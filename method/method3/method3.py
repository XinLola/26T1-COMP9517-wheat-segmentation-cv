import json
import csv
import time
import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib



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


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    # Internal convention: pred_mask > 0  means predicted as plant (foreground)
    # EWS dataset convention: gt_mask == 0 (black) means plant (foreground)
    pred = (pred_mask > 0)    # True = predicted as plant
    gt = (gt_mask == 0)       # True = actually plant

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

    # Display convention: black (0) = plant for both pred and gt (matches EWS saved format)
    pred_display = np.where(pred_mask > 0, 0, 255).astype(np.uint8)  # invert internal: 255->0 (plant)
    pred_rgb = cv2.cvtColor(pred_display, cv2.COLOR_GRAY2RGB)

    gt_display = np.where(gt_mask == 0, 0, 255).astype(np.uint8)     # gt already: 0=plant, keep as-is
    gt_rgb = cv2.cvtColor(gt_display, cv2.COLOR_GRAY2RGB)

    panels = [img_rgb, pred_rgb, gt_rgb]
    titles = ["Original image", "Prediction mask", "Actual mask"]

    h, w = img_rgb.shape[:2]
    combined = []
    for panel, title in zip(panels, titles):
        # add a title bar on top
        canvas = np.full((h + 40, w, 3), 255, dtype=np.uint8)
        canvas[40:, :, :] = panel
        cv2.putText(canvas, title, (15, 27),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        combined.append(canvas)

    result = np.hstack(combined)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def compute_lbp(gray, x, y):
 
    h, w = gray.shape
    centre = gray[y, x]
    pattern = 0

    #8 neighbours in clockwise order starting from top-left
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0),
               (1, -1), (0, -1)]

    for i, (dy, dx) in enumerate(offsets):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            if gray[ny, nx] >= centre:
                pattern |= (1 << i)
        # if out of boundary, treat as 0 (bit stays 0)

    return pattern


def compute_lbp_image(gray):
 
    h, w = gray.shape
    # Use int16 to avoid overflow during the 8-step bit accumulation (max value = 255)
    lbp = np.zeros((h, w), dtype=np.int16)

    #pad the image with zeros on all sides
    padded = np.pad(gray, 1, mode='constant', constant_values=0)

    # 8 neighbour offsets (row_offset, col_offset) clockwise from top-left
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0),
               (1, -1), (0, -1)]

    centre = padded[1:h+1, 1:w+1].astype(np.int16)

    for i, (dy, dx) in enumerate(offsets):
        neighbour = padded[1+dy:h+1+dy, 1+dx:w+1+dx].astype(np.int16)
        lbp += ((neighbour >= centre).astype(np.int16)) << i

    return lbp.astype(np.uint8)  # safe: accumulated value is in [0, 255]



def extract_features(img_bgr: np.ndarray) -> np.ndarray:

    h, w = img_bgr.shape[:2]

    # normalise RGB to [0, 1]
    img_float = img_bgr.astype(np.float32) / 255.0
    b_ch = img_float[:, :, 0]
    g_ch = img_float[:, :, 1]
    r_ch = img_float[:, :, 2]

    #convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_float = hsv.astype(np.float32)
    hsv_float[:, :, 0] /= 180.0   # H range is 0-180 in OpenCV
    hsv_float[:, :, 1] /= 255.0
    hsv_float[:, :, 2] /= 255.0

    # vegetation indices
    exg = 2.0 * g_ch - r_ch - b_ch    # excess green
    exr = 1.4 * r_ch - g_ch        # excess red
    exg_minus_exr = exg - exr

    #local statistics of green channel using box filter
    kernel_size = 5
    g_uint8 = (g_ch * 255).astype(np.uint8)
    local_mean = cv2.blur(g_ch, (kernel_size, kernel_size))

    # local std: std = sqrt(mean(x^2) - mean(x)^2)
    g_sq = g_ch ** 2
    local_mean_sq = cv2.blur(g_sq, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    # gaussian blur as a smoothed feature
    g_blurred = cv2.GaussianBlur(g_ch, (5, 5), sigmaX=1.0)

    # LBP texture
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = compute_lbp_image(gray)
    lbp_norm = lbp.astype(np.float32) / 255.0

    # stack all features: shape (H, W, 13)
    features = np.stack([
        r_ch, g_ch, b_ch,# RGB (3)
        hsv_float[:, :, 0], # H (1)
        hsv_float[:, :, 1], # S (1)
        hsv_float[:, :, 2], # V (1)
        exg, # ExG (1)
        exr, # ExR (1)
        exg_minus_exr, # ExG - ExR (1)
        local_mean, # local mean of G (1)
        local_std, # local std of G (1)
        g_blurred, # smoothed G (1)
        lbp_norm,                           # LBP (1)
    ], axis=-1)

    # reshape to (num_pixels, num_features)
    features_2d = features.reshape(-1, features.shape[-1])
    return features_2d


def sample_pixels(features_2d, labels_1d, samples_per_image=2000, random_seed=42):
 
    rng = np.random.RandomState(random_seed)
    n_pixels = features_2d.shape[0]

    if n_pixels <= samples_per_image:
        return features_2d, labels_1d

    indices = rng.choice(n_pixels, size=samples_per_image, replace=False)
    return features_2d[indices], labels_1d[indices]


def train_random_forest(train_dir: Path, val_dir: Path,
                        samples_per_image=2000, random_seed=42):

    print("---- Training Phase “---")

    # collect training data
    train_pairs = image_to_mask(train_dir)
    print(f"Found {len(train_pairs)} training image-mask pairs")

    all_features = []
    all_labels = []

    for i, (img_path, gt_path) in enumerate(train_pairs):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        #extract feature
        feats = extract_features(img_bgr)

        # EWS dataset convention: gt_mask == 0 (black) = plant → label 1 (foreground)
        labels = (gt_mask.flatten() == 0).astype(np.int32)

        # sample to keep training size reasonable
        sampled_feats, sampled_labels = sample_pixels(
            feats, labels,
            samples_per_image=samples_per_image,
            random_seed=random_seed + i
        )

        all_features.append(sampled_feats)
        all_labels.append(sampled_labels)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(train_pairs)} training images...")

    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_labels)
    print(f"Training set size: {X_train.shape[0]} pixels, {X_train.shape[1]} features")
    print(f"  Plant pixels: {(y_train == 1).sum()}, Soil pixels: {(y_train == 0).sum()}")

    # collect validation data (use all pixels for proper evaluation) 
    val_pairs = image_to_mask(val_dir)
    print(f"\nFound {len(val_pairs)} validation image-mask pairs")

    val_features = []
    val_labels = []

    for img_path, gt_path in val_pairs:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        feats = extract_features(img_bgr)
        # EWS dataset convention: gt_mask == 0 (black) = plant → label 1 (foreground)
        labels = (gt_mask.flatten() == 0).astype(np.int32)
        # sample validation too (otherwise it's too many pixels)
        sampled_feats, sampled_labels = sample_pixels(
            feats, labels,
            samples_per_image=3000,
            random_seed=random_seed
        )
        val_features.append(sampled_feats)
        val_labels.append(sampled_labels)

    X_val = np.vstack(val_features)
    y_val = np.concatenate(val_labels)
    print(f"Validation set size: {X_val.shape[0]} pixels")

    #grid search for hyperparameters
    print("\n=== Hyperparameter Tuning ===")

    # trying a small grid of parameters for speed
    param_grid = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 100, "max_depth": 20},
        {"n_estimators": 150, "max_depth": 20},
    ]

    best_model = None
    best_iou = 0.0
    best_params = None

    for params in param_grid:
        print(f"\n  Trying: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")

        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_seed,
            n_jobs=-1,           # use all CPU cores
            class_weight="balanced"  # handle class imbalance
        )
        clf.fit(X_train, y_train)

        # evaluate on validation set
        y_pred = clf.predict(X_val)
        tp = np.logical_and(y_pred == 1, y_val == 1).sum()
        fp = np.logical_and(y_pred == 1, y_val == 0).sum()
        fn = np.logical_and(y_pred == 0, y_val == 1).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        print(f"    Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, IoU={iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            best_model = clf
            best_params = params

    print(f"\nBest params: {best_params}, Best val IoU: {best_iou:.4f}")
    return best_model


def predict_image(model, img_bgr: np.ndarray) -> np.ndarray:
    
    h, w = img_bgr.shape[:2]

    # extract features
    features = extract_features(img_bgr)

    # predict
    pred_labels = model.predict(features)

    # Internal convention: label 1 = plant (foreground) → mapped to 255
    pred_mask = pred_labels.reshape(h, w).astype(np.uint8) * 255  # 255 = plant

    # optional: clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    # remove small connected components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    cleaned = np.zeros_like(pred_mask)
    min_area = 100
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    pred_mask = cleaned

    return pred_mask


def apply_distortion(img_bgr, kind, seed=0):
    import numpy as np
    import cv2
    rng = np.random.default_rng(seed)
    img = img_bgr.astype(np.float32)
    k = kind.lower()
    if k == "gaussian_noise":
        noise = rng.normal(loc=0.0, scale=12.75, size=img.shape).astype(np.float32)
        out = np.clip(img + noise, 0, 255)
    elif k == "gaussian_blur":
        out = cv2.GaussianBlur(img, (5, 5), sigmaX=1.2)
    elif k == "low_brightness":
        out = np.clip(img * 0.60, 0, 255)
    elif k == "low_contrast":
        img_norm = img / 255.0
        out_norm = np.clip(np.power(img_norm, 1.25) * 0.8, 0.0, 1.0)
        out = out_norm * 255.0
    elif k == "occlusion":
        out = img.copy()
        h, w, _ = out.shape
        occ_h = max(1, int(h * 0.20))
        occ_w = max(1, int(w * 0.20))
        y0 = int(rng.integers(0, max(h - occ_h + 1, 1)))
        x0 = int(rng.integers(0, max(w - occ_w + 1, 1)))
        out[y0 : y0 + occ_h, x0 : x0 + occ_w, :] = 0.0
    else:
        out = img
    return out.astype(np.uint8)

def _failure_reason(metric):
    p = float(metric["precision"])
    r = float(metric["recall"])
    iou = float(metric["iou"])
    if r < 0.55 and p >= 0.70:
        return "Under-segmentation: Handcrafted features failed to separate plant under this lighting."
    if p < 0.55 and r >= 0.70:
        return "Over-segmentation: RF misclassified background textures (like soil/shadow) as plant."
    if iou < 0.40 and abs(p - r) < 0.15:
        return "Total confusion: Overall feature breakdown on this image."
    return "Complex failure: Handcrafted LBP/HSV features are insufficient here."

def _save_failure_overlay(image_bgr, pred, gt, out_path):
    import numpy as np
    import cv2
    base = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pred_b = pred > 0
    gt_b = gt == 0
    overlay = base.copy()
    fp = np.logical_and(pred_b, np.logical_not(gt_b))
    fn = np.logical_and(np.logical_not(pred_b), gt_b)
    tp = np.logical_and(pred_b, gt_b)
    overlay[tp] = 0.6 * overlay[tp] + 0.4 * np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    overlay[fp] = 0.5 * overlay[fp] + 0.5 * np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[fn] = 0.5 * overlay[fn] + 0.5 * np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor((overlay * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))

def _write_csv(path, rows):
    import csv
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main(train_dir, val_dir, test_dir, output_root, samples_per_image=2000):
    import time
    import json
    import numpy as np
    import cv2
    import joblib
    masks_dir = output_root / "masks_final"
    visuals_dir = output_root / "visuals"
    failure_dir = output_root / "failure_cases"
    model_path = output_root / "rf_model.joblib"

    masks_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    model = train_random_forest(train_dir, val_dir, samples_per_image=samples_per_image)
    train_time = time.perf_counter() - train_start
    print(f"\nTotal training time: {train_time:.2f} seconds")

    joblib.dump(model, str(model_path))

    print("\n=== Testing Phase ===")
    test_pairs = image_to_mask(test_dir)
    details = []
    
    for img_path, gt_path in test_pairs:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        pred_mask = predict_image(model, img_bgr)
        elapsed = time.perf_counter() - start
        
        cv2.imwrite(str(masks_dir / img_path.name), cv2.bitwise_not(pred_mask))
        vis = visuals(img_bgr, pred_mask, gt_mask)
        cv2.imwrite(str(visuals_dir / img_path.name), vis)
        
        m = compute_metrics(pred_mask, gt_mask)
        details.append({"sample_id": img_path.stem, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "iou": m["iou"], "test_time_sec": elapsed})
        
    def _summary(dlist):
        out = {}
        for k in ["precision", "recall", "f1", "iou", "test_time_sec"]:
            vals = [d[k] for d in dlist]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
        return out
        
    baseline_summary = _summary(details)
    
    print("[INFO] Running Robustness experiments ...")
    robustness_rows = []
    for kind in ("gaussian_noise", "gaussian_blur", "low_brightness", "low_contrast", "occlusion"):
        rob_details = []
        for i, (img_path, gt_path) in enumerate(test_pairs):
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            distorted = apply_distortion(img_bgr, kind, seed=42+i)
            
            start = time.perf_counter()
            pred_mask = predict_image(model, distorted)
            elapsed = time.perf_counter() - start
            
            m = compute_metrics(pred_mask, gt_mask)
            rob_details.append({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "iou": m["iou"], "test_time_sec": elapsed})
        
        r_sum = _summary(rob_details)
        robustness_rows.append({"distortion": kind, "precision": r_sum["precision_mean"], "recall": r_sum["recall_mean"], "f1": r_sum["f1_mean"], "iou": r_sum["iou_mean"], "test_time_mean_sec": r_sum["test_time_sec_mean"]})
        
    print("[INFO] Generating Failure Analysis ...")
    worst = sorted(details, key=lambda x: x["iou"])[:5]
    failure_lines = ["# Failure Analysis (Method 3)", "", "| sample_id | precision | recall | f1 | iou | reason |", "|---|---:|---:|---:|---:|---|"]
    for item in worst:
        sid = item["sample_id"]
        pair = next(p for p in test_pairs if p[0].stem == sid)
        img_bgr = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(pair[1]), cv2.IMREAD_GRAYSCALE)
        pred_mask = predict_image(model, img_bgr)
        _save_failure_overlay(img_bgr, pred_mask, gt_mask, failure_dir / f"{sid}.png")
        reason = _failure_reason(item)
        failure_lines.append(f"| {sid} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['iou']:.4f} | {reason} |")

    (output_root / "failure_analysis.md").write_text("\n".join(failure_lines), encoding="utf-8")
    
    _write_csv(output_root / "robustness_results.csv", robustness_rows)
    _write_csv(output_root / "test_details_final.csv", details)
    
    write_json = lambda p, d: p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(output_root / "evaluation_summary.json", {
        "method": "Method 3 (ML: Random Forest + Features)",
        "type": "ml",

        "train": {
            "train_time_sec": train_time,
            "params": {
                "samples_per_image": samples_per_image
            }
        },

        "validation": None,

        "test": {
            "baseline": baseline_summary,
            "robustness": robustness_rows,
            "extra": {}
        },

        "timing": {
            "train_time_sec": train_time,
            "test_time_mean_sec": baseline_summary["test_time_sec_mean"]
        }
    })
    print(f"[INFO] Method 3 complete. Results saved to {output_root}")

if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    train_path = project_root / "data" / "train"
    val_path = project_root / "data" / "validation"
    test_path = project_root / "data" / "test"
    output = project_root / "output" / "method3"
    main(train_dir=train_path, val_dir=val_path, test_dir=test_path, output_root=output)
