import json
import time
import cv2
import numpy as np
from pathlib import Path
import csv

def image_to_mask(split_dir: Path):
    f = sorted([p for p in split_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
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

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
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

def apply_distortion(img_bgr: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
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

def _failure_reason(metric: dict) -> str:
    p = float(metric["precision"])
    r = float(metric["recall"])
    iou = float(metric["iou"])
    if r < 0.55 and p >= 0.70:
        return "Under-segmentation: Lighting or background prevented foreground extraction."
    if p < 0.55 and r >= 0.70:
        return "Over-segmentation: Unsupervised GrabCut might misclassify dark soil as plant."
    if iou < 0.40 and abs(p - r) < 0.15:
        return "Total confusion: ExG greenness feature failed on this image."
    return "Complex failure: Insufficient contrast between background and foreground greenness."

def _save_failure_overlay(image_bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray, out_path: Path) -> None:
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

def _write_csv(path: Path, rows: list) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def predict_method1(img_bgr: np.ndarray, grabcut_iterations: int = 6, erosion_size: int = 6, dilation_size: int = 6, min_area: int = 150):
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

    cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, grabcut_iterations, cv2.GC_INIT_WITH_MASK)

    pred_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel_open)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    cleaned = np.zeros_like(pred_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned

def main(split_dir: Path, output_root: Path):
    masks_dir = output_root / "masks_final"
    visuals_dir = output_root / "visuals"
    failure_dir = output_root / "failure_cases"
    masks_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = image_to_mask(split_dir)

    if len(pairs) == 0:
        raise ValueError(f"No data found in {split_dir}")

    print(f"[INFO] Found {len(pairs)} test images.")
    print("[INFO] Evaluating Baseline on Test set ...")
    
    details = []
    
    # Baseline Eval
    print("[INFO] Evaluating Baseline on Test set ...")
    for i, (img_path, gt_path) in enumerate(pairs):
        print(f"[INFO] {i+1}/{len(pairs)} -> {img_path.name}")

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        start = time.perf_counter()
        pred_mask = predict_method1(img_bgr, grabcut_iterations=3)  
        elapsed = time.perf_counter() - start
        
        cv2.imwrite(str(masks_dir / img_path.name), cv2.bitwise_not(pred_mask))
        vis = visuals(img_bgr, pred_mask, gt_mask)
        cv2.imwrite(str(visuals_dir / img_path.name), vis)

        m = compute_metrics(pred_mask, gt_mask)

        # save failure cases
        if m["iou"] < 0.40:
            reason = _failure_reason(m)
            _save_failure_overlay(
                img_bgr, pred_mask, gt_mask,
                failure_dir / f"{img_path.stem}_failure.png"
            )
            print(f"    ⚠️ Failure case detected! {reason}")

        details.append({
            "sample_id": img_path.stem,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "iou": m["iou"],
            "test_time_sec": elapsed
        })
        
    def _summary(dlist):
        out = {}
        for k in ["precision", "recall", "f1", "iou", "test_time_sec"]:
            vals = [d[k] for d in dlist]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
        return out
        
    baseline_summary = _summary(details)
    
    # Robustness Experiments
    print("[INFO] Running Robustness experiments ...")
    robustness_rows = []
    for kind in ("gaussian_noise", "gaussian_blur", "low_brightness", "low_contrast", "occlusion"):
        print(f"       -> {kind}")
        rob_details = []
        for i, (img_path, gt_path) in enumerate(pairs):
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            distorted = apply_distortion(img_bgr, kind, seed=42+i)
            
            start = time.perf_counter()
            pred_mask = predict_method1(distorted)
            elapsed = time.perf_counter() - start
            
            m = compute_metrics(pred_mask, gt_mask)
            rob_details.append({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "iou": m["iou"], "test_time_sec": elapsed})
        
        r_sum = _summary(rob_details)
        robustness_rows.append({
            "distortion": kind,
            "precision": r_sum["precision_mean"],
            "recall": r_sum["recall_mean"],
            "f1": r_sum["f1_mean"],
            "iou": r_sum["iou_mean"],
            "test_time_mean_sec": r_sum["test_time_sec_mean"]
        })
        
    # Failure Analysis
    print("[INFO] Generating Failure Analysis ...")
    worst = sorted(details, key=lambda x: x["iou"])[:5]
    failure_lines = [
        "# Failure Analysis (Method 1)",
        "",
        "| sample_id | precision | recall | f1 | iou | reason |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in worst:
        sid = item["sample_id"]
        pair = next(p for p in pairs if p[0].stem == sid)
        img_bgr = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(pair[1]), cv2.IMREAD_GRAYSCALE)
        pred_mask = predict_method1(img_bgr)
        _save_failure_overlay(img_bgr, pred_mask, gt_mask, failure_dir / f"{sid}.png")
        
        reason = _failure_reason(item)
        failure_lines.append(f"| {sid} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['iou']:.4f} | {reason} |")

    (output_root / "failure_analysis.md").write_text("\n".join(failure_lines), encoding="utf-8")
    
    # Save Outputs
    print("[INFO] Saving artifacts ...")
    _write_csv(output_root / "robustness_results.csv", robustness_rows)
    _write_csv(output_root / "test_details_final.csv", details)
    
    write_json = lambda path, payload: path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    write_json(output_root / "evaluation_summary.json", {
        "method": "Method 1 (Unsupervised: ExG + GrabCut)",
        "type": "unsupervised",

        "train": None,
        "validation": None,

        "test": {
            "baseline": baseline_summary,
            "robustness": robustness_rows,
            "extra": {}
        },

        "timing": {
            "train_time_sec": None,
            "test_time_mean_sec": baseline_summary["test_time_sec_mean"]
        }
    })
    
    print(f"[INFO] Method 1 complete. Results saved to {output_root}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    path = project_root / "data" / "test"
    output = project_root / "output" / "method1"
    main(split_dir=path, output_root=output)