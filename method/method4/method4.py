import json
import time
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# same as previous method's image_to_mask
def image_to_mask(split_dir: Path):

    files = sorted([p for p in split_dir.iterdir() if p.suffix.lower() == ".png"])
    mask_dict = {}
    image_list = []
    for p in files:
        if p.stem.endswith("_mask"):
            base_name = p.stem.replace("_mask", "")
            mask_dict[base_name] = p
        else:
            image_list.append(p)
    pairs = []
    for img_path in image_list:
        if img_path.stem in mask_dict:
            pairs.append((img_path, mask_dict[img_path.stem]))
    return pairs



# same evaluation with previous method,
def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    # Internal convention: pred_mask > 0  means predicted as plant (foreground)
    # EWS dataset convention: gt_mask == 0 (black) means plant (foreground)
    pred = (pred_mask > 0)
    gt = (gt_mask == 0)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def visuals(img_bgr: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Display convention: black (0) = plant for both pred and gt
    pred_display = np.where(pred_mask > 0, 0, 255).astype(np.uint8)
    pred_rgb = cv2.cvtColor(pred_display, cv2.COLOR_GRAY2RGB)

    gt_display = np.where(gt_mask == 0, 0, 255).astype(np.uint8)
    gt_rgb = cv2.cvtColor(gt_display, cv2.COLOR_GRAY2RGB)

    panels = [img_rgb, pred_rgb, gt_rgb]
    titles = ["Original image", "Prediction mask", "Actual mask"]

    h, w = img_rgb.shape[:2]
    combined = []
    for panel, title in zip(panels, titles):
        canvas = np.full((h + 40, w, 3), 255, dtype=np.uint8)
        canvas[40:, :, :] = panel
        cv2.putText(canvas, title, (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        combined.append(canvas)

    result = np.hstack(combined)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)



#dataset class for loading image-mask pairs and applying during training
class WheatSegDataset(Dataset):

    def __init__(self, pairs, augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, img_path: Path, gt_path: Path):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_mask is None:
            raise FileNotFoundError(f"Failed to get {img_path} or {gt_path}")

        #convert to RGB float tensor and normalise to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # EWS convention: gt == 0 (black) means plant -> label 1
        label = (gt_mask == 0).astype(np.float32)
        return img_rgb, label

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        #Random horizontal flip
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        # Random vertical flip
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        # Random 90 degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        # Mild brightness jitter
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0.0, 1.0)
        return img, mask

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        img, mask = self._load(img_path, gt_path)
        if self.augment:
            img, mask = self._augment(img, mask)
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return img_tensor, mask_tensor


# U-net arch with 4 down sampling steps, batch norm, and skip connections.
#Base channel count is 32.
class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        #Pad input the multiple of 16. 
        #after out of net, will crop back to original size
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)

        #Crop back to the original (un-padded) spatial size
        return out[:, :, :h, :w]




# Loss functions:
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # logits: (B, 1, H, W); target: (B, 1, H, W) in {0, 1}
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    return 0.5 * bce + 0.5 * dice

# IOU evaluation on validation set
@torch.no_grad()
def evaluate_iou(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    tp_total, fp_total, fn_total = 0, 0, 0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()

        tp = ((preds == 1) & (masks == 1)).sum().item()
        fp = ((preds == 1) & (masks == 0)).sum().item()
        fn = ((preds == 0) & (masks == 1)).sum().item()
        tp_total += tp
        fp_total += fp
        fn_total += fn

    denom = tp_total + fp_total + fn_total
    return tp_total / denom if denom > 0 else 0.0




# how to training loop for U-Net model
def train_unet(train_dir: Path, val_dir: Path,
               epochs: int = 25,
               batch_size: int = 8,
               lr: float = 1e-3,
               device: torch.device = None,
               random_seed: int = 42):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("************** Training Phase (U-Net model) **************")
    print(f"Device: {device}")

    train_pairs = image_to_mask(train_dir)
    val_pairs = image_to_mask(val_dir)
    print(f"Found {len(train_pairs)} training and {len(val_pairs)} validation pairs")

    train_ds = WheatSegDataset(train_pairs, augment=True)
    val_ds = WheatSegDataset(val_pairs, augment=False)

    # training: batch_size =8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    # test only: batch_size =2
    # train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
    #                       num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    model = UNet(in_ch=3, out_ch=1, base=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running / max(n_batches, 1)

        val_iou = evaluate_iou(model, val_loader, device)
        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | val IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation IoU: {best_iou:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


# inference stage
@torch.no_grad()
def predict_image(model: nn.Module, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Internal convention: 255 = plant (foreground)
    pred_mask = (probs > 0.5).astype(np.uint8) * 255

    # Light morphological cleanup (consistent with Method 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    return pred_mask


#main and summary generation
def main(train_dir: Path, val_dir: Path, test_dir: Path, output_root: Path,
         epochs: int = 25, batch_size: int = 8, lr: float = 1e-3):

    masks_dir = output_root / "masks"
    visuals_dir = output_root / "visuals"
    metrics_path = output_root / "metrics.json"
    model_path = output_root / "unet_model.pt"

    masks_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    model, device = train_unet(train_dir, val_dir,
                               epochs=epochs, batch_size=batch_size, lr=lr)
    train_time = time.perf_counter() - train_start
    print(f"\nTotal training time: {train_time:.2f} seconds")

    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to {model_path}")

    # ---- Testing ----
    print("\n************** Testing Phase **************")
    test_pairs = image_to_mask(test_dir)
    print(f"Found {len(test_pairs)} test image-mask pairs")

    results = []
    total_predict_time = 0.0

    for img_path, gt_path in test_pairs:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise FileNotFoundError(f"Cannot read mask: {gt_path}")

        start = time.perf_counter()
        pred_mask = predict_image(model, img_bgr, device)
        elapsed = time.perf_counter() - start
        total_predict_time += elapsed

        # Save using EWS convention: 0 (black) = plant
        cv2.imwrite(str(masks_dir / img_path.name), cv2.bitwise_not(pred_mask))

        metrics = compute_metrics(pred_mask, gt_mask)
        vis = visuals(img_bgr, pred_mask, gt_mask)
        cv2.imwrite(str(visuals_dir / img_path.name), vis)

        results.append({
            "image_name": img_path.name,
            "predict_time": elapsed,
            **metrics,
        })

        print(f"  {img_path.name}: F1={metrics['f1']:.4f}, "
              f"IoU={metrics['iou']:.4f}, Time={elapsed:.3f}s")

    summary = {
        "method": "U-Net (deep learning)",
        "backbone": "Custom U-Net, base=32, BCE+Dice loss, Adam + cosine LR",
        "augmentation": "random flips, random 90-deg rotations, brightness jitter",
        "training_time": train_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "metrics": {
            "precision": float(np.mean([r["precision"] for r in results])),
            "recall": float(np.mean([r["recall"] for r in results])),
            "f1": float(np.mean([r["f1"] for r in results])),
            "iou": float(np.mean([r["iou"] for r in results])),
            "average_predict_time_per_image": total_predict_time / len(test_pairs),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Results Summary ===")
    print(f"  Average Precision: {summary['metrics']['precision']:.4f}")
    print(f"  Average Recall:    {summary['metrics']['recall']:.4f}")
    print(f"  Average F1:        {summary['metrics']['f1']:.4f}")
    print(f"  Average IoU:       {summary['metrics']['iou']:.4f}")
    print(f"  Avg predict time:  {summary['metrics']['average_predict_time_per_image']:.3f}s")
    print(f"\nResults saved to {output_root}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    train_path = project_root / "data" / "train"
    val_path = project_root / "data" / "validation"
    test_path = project_root / "data" / "test"
    output = project_root / "output" / "method4"

    # Training
    main(train_dir=train_path, val_dir=val_path,
         test_dir=test_path, output_root=output,
         epochs=25, batch_size=8, lr=1e-3)

    # Testing only
    # main(train_dir=train_path, val_dir=val_path,
    #      test_dir=test_path, output_root=output,
    #      epochs=1, batch_size =2, lr=1e-3)
