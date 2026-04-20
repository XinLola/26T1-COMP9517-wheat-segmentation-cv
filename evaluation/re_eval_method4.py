from pathlib import Path
import cv2
import numpy as np
import json
import time


def compute_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0)
    gt = (gt_mask == 0)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return precision, recall, f1, iou


def main():
    root = Path(__file__).resolve().parent.parent
    output_root = root / "output" / "method4"
    test_root = root / "data" / "test"

    preds_dir = output_root / "masks_final"

    # pairing
    images = sorted([p for p in test_root.iterdir() if not p.stem.endswith("_mask")])
    masks = {p.stem.replace("_mask", ""): p for p in test_root.iterdir() if p.stem.endswith("_mask")}

    results = []

    total_time = 0.0

    for img_path in images:
        if img_path.stem not in masks:
            continue

        gt_path = masks[img_path.stem]
        pred_path = preds_dir / img_path.name

        if not pred_path.exists():
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

        start = time.perf_counter()
        p, r, f1, iou = compute_metrics(pred, gt)
        elapsed = time.perf_counter() - start

        total_time += elapsed

        results.append({
            "precision": p,
            "recall": r,
            "f1": f1,
            "iou": iou,
            "test_time_sec": elapsed
        })

    # summary
    def mean_std(key):
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for k in ["precision", "recall", "f1", "iou", "test_time_sec"]:
        m, s = mean_std(k)
        summary[f"{k}_mean"] = m
        summary[f"{k}_std"] = s

    final = {
        "method": "Method 4 (U-Net)",
        "test": {
            "baseline": summary
        },
        "timing": {
            "test_time_mean_sec": summary["test_time_sec_mean"]
        }
    }

    out_path = output_root / "evaluation_summary.json"
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")

    print(f"[OK] Method4 re-evaluated → {out_path}")


if __name__ == "__main__":
    main()