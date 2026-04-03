import json
import time
import cv2
import numpy as np
from pathlib import Path

def image_to_mask(split_dir: Path):
    f = sorted([p for p in split_dir.iterdir() if p.suffix.lower() == ".png"])
    mask = {}
    image = []
    for p in f:
        if p.stem.endswith("_mask"):
            mask[p.stem.replace("_mask", "")] = p
        else:
            image.append(p)
    both = []
    for img in image:
        if img.stem in mask:
            both.append((img, mask[img.stem]))
    return both

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
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

def visuals(img_bgr: np.ndarray, pred_mask: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pred_binary = np.where(pred_mask > 0, 0, 255).astype(np.uint8)
    pred_rgb = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2RGB)
    binary = np.where(mask == 0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    panels = [img_rgb, pred_rgb, rgb]
    titles = ["Original image", "Prediction mask", "Actual mask"]
    h, w = img_rgb.shape[:2]
    t = []
    for panel, title in zip(panels, titles):
        canvas = np.full((h + 40, w, 3), 255, dtype=np.uint8)
        canvas[40:, :, :] = panel
        cv2.putText(canvas, title, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        t.append(canvas)
    result = np.hstack(t)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
def main(split_dir: Path, output_root: Path, grabcut_iterations: int = 6, erosion_size: int = 6, dilation_size: int = 6, min_area: int = 150, ):
    masks_dir = output_root / "masks"
    visuals_dir = output_root / "visuals"
    metrics_path = output_root / "metrics.json"
    masks_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    pairs = image_to_mask(split_dir)
    results = []
    total_time = 0.0
    for img_path, gt_path in pairs:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise FileNotFoundError(f"Cannot read mask: {gt_path}")
        start = time.perf_counter()
        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)
        exg = 2 * g - r - b
        exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, coarse_fg = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = coarse_fg.shape
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

        sure_fg = cv2.erode(coarse_fg, erode_kernel, iterations=1)
        probable_fg = coarse_fg.copy()
        probable_bg = cv2.bitwise_not(cv2.dilate(coarse_fg, dilate_kernel, iterations=1))

        gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
        gc_mask[probable_bg > 0] = cv2.GC_PR_BGD
        gc_mask[sure_fg > 0] = cv2.GC_FGD

        border = max(5, min(h, w) // 35)
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            grabcut_iterations,
            cv2.GC_INIT_WITH_MASK
        )

        pred_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel_open)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
        cleaned = np.zeros_like(pred_mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == label] = 255
        pred_mask = cleaned

        elapsed = time.perf_counter() - start
        total_time += elapsed

        cv2.imwrite(str(masks_dir / img_path.name), cv2.bitwise_not(pred_mask))

        metrics = compute_metrics(pred_mask, gt_mask)

        visual = visuals(img_bgr, pred_mask, gt_mask)
        cv2.imwrite(str(visuals_dir / img_path.name), visual)

        results.append({
            "image_name": img_path.name,
            "predict_time": elapsed,
            **metrics,
        })

    summary = {
        "metrics": {
            "precision": float(np.mean([r["precision"] for r in results])),
            "recall": float(np.mean([r["recall"] for r in results])),
            "f1": float(np.mean([r["f1"] for r in results])),
            "iou": float(np.mean([r["iou"] for r in results])),
            "average_predict_time_per_image": total_time / len(pairs),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
path = Path("data/test")
output = Path("output/method1")
main(split_dir=path, output_root=output)