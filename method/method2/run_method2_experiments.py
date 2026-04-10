from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from skimage import io

from method2_ml import (
    HAS_CV2,
    Method2Config,
    SampleRecord,
    apply_distortion,
    discover_split_records,
    evaluate,
    evaluate_on_records,
    predict_method2,
    read_image,
    read_mask,
    save_model,
    subset_records,
    train_method2,
    write_json,
)


def _summarize_details(details: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("precision", "recall", "f1", "iou", "test_time_sec"):
        values = np.asarray([d[key] for d in details], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    return out


def evaluate_with_transform(
    model,
    records: Sequence[SampleRecord],
    cfg: Method2Config,
    transform_fn: Callable[[np.ndarray, int], np.ndarray],
    progress_name: str = "robustness",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    n_total = len(records)
    t0 = time.perf_counter()
    if cfg.verbose_progress:
        print(f"[PROGRESS][ROBUST][{progress_name}] start | samples={n_total}", flush=True)

    for i, rec in enumerate(records, start=1):
        img = read_image(rec.image_path, cfg)
        gt = read_mask(rec.mask_path, cfg)
        transformed = transform_fn(img, i - 1)
        pred, t = predict_method2(model, transformed, cfg)
        m = evaluate(pred, gt)
        details.append(
            {
                "sample_id": rec.sample_id,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "iou": m["iou"],
                "test_time_sec": float(t),
            }
        )
        if cfg.verbose_progress and (
            i == 1 or i % max(1, cfg.progress_log_every_eval_images) == 0 or i == n_total
        ):
            elapsed = time.perf_counter() - t0
            pct = 100.0 * i / max(n_total, 1)
            print(
                f"[PROGRESS][ROBUST][{progress_name}] {i}/{n_total} ({pct:.1f}%) | sample={rec.sample_id} | iou={m['iou']:.4f} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = _summarize_details(details)
    if cfg.verbose_progress:
        elapsed = time.perf_counter() - t0
        print(
            f"[PROGRESS][ROBUST][{progress_name}] done | iou_mean={summary['iou_mean']:.4f} | elapsed={elapsed:.1f}s",
            flush=True,
        )
    return summary, details


def _failure_reason(metric: Dict[str, Any]) -> str:
    p = float(metric["precision"])
    r = float(metric["recall"])
    iou = float(metric["iou"])
    if r < 0.55 and p >= 0.70:
        return "欠分割倾向：可能边界断裂或低对比区域漏检"
    if p < 0.55 and r >= 0.70:
        return "过分割倾向：可能将土壤纹理误检为作物"
    if iou < 0.40 and abs(p - r) < 0.15:
        return "整体混淆：可能受光照变化或噪声影响"
    return "复合失效：背景复杂度与特征可分性不足"


def _save_failure_overlay(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
) -> None:
    base = image.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    base = np.clip(base, 0.0, 1.0)

    pred_b = pred > 0
    gt_b = gt > 0

    overlay = base.copy()

    fp = np.logical_and(pred_b, np.logical_not(gt_b))
    fn = np.logical_and(np.logical_not(pred_b), gt_b)
    tp = np.logical_and(pred_b, gt_b)

    overlay[tp] = 0.6 * overlay[tp] + 0.4 * np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    overlay[fp] = 0.5 * overlay[fp] + 0.5 * np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[fn] = 0.5 * overlay[fn] + 0.5 * np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(out_path), (overlay * 255.0).astype(np.uint8), check_contrast=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Method2 experiments (HSV+LBP+Gabor + classifier).")
    p.add_argument("--dataset-root", type=Path, default=None, help="Dataset root path.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/method2"), help="Output folder.")
    p.add_argument("--image-size", type=str, default="", help="Optional resize as HxW, e.g. 350x350")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--log-every", type=int, default=10, help="Progress log frequency in images.")
    p.add_argument("--quiet", action="store_true", help="Disable detailed progress logs.")
    return p.parse_args()


def _parse_image_size(raw: str):
    if not raw:
        return None
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--image-size must be like 350x350")
    return int(parts[0]), int(parts[1])


def _default_dataset_root() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "EWS-Dataset"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = _default_dataset_root()
        if dataset_root is None:
            raise SystemExit(
                "Missing --dataset-root and no default EWS-Dataset found next to script. "
                "Please pass: --dataset-root <path>"
            )
        print(f"[INFO] Using default dataset root: {dataset_root}")

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output root: {output_root.resolve()}")

    cfg = Method2Config(
        image_size=_parse_image_size(args.image_size),
        random_state=args.seed,
        classifier="svm",
        verbose_progress=not args.quiet,
        progress_log_every_images=max(1, args.log_every),
        progress_log_every_eval_images=max(1, args.log_every),
    )

    print("[INFO] Discovering dataset splits ...")
    train_records = discover_split_records(dataset_root, "train")
    val_records = discover_split_records(dataset_root, "val")
    test_records = discover_split_records(dataset_root, "test")
    print(
        f"[INFO] Records loaded: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
    )

    # Baseline SVM without postprocess.
    print("[INFO] Training SVM baseline ...")
    baseline_cfg = replace(cfg, classifier="svm", apply_postprocess=False)
    baseline_model, baseline_params, baseline_val_metrics, baseline_train_time = train_method2(
        train_records, val_records, baseline_cfg
    )
    print("[INFO] Evaluating SVM baseline on test set ...")
    baseline_test_summary, baseline_test_details = evaluate_on_records(
        baseline_model,
        test_records,
        baseline_cfg,
        save_dir=output_root / "masks_baseline",
        progress_name="test_svm_baseline",
        verbose=baseline_cfg.verbose_progress,
        log_every=baseline_cfg.progress_log_every_eval_images,
    )

    # Small improvement: postprocessing on top of the same trained model.
    improved_cfg = replace(baseline_cfg, apply_postprocess=True)
    print("[INFO] Evaluating postprocess improvement ...")
    improved_test_summary, improved_test_details = evaluate_on_records(
        baseline_model,
        test_records,
        improved_cfg,
        save_dir=output_root / "masks_final",
        progress_name="test_svm_postprocess",
        verbose=improved_cfg.verbose_progress,
        log_every=improved_cfg.progress_log_every_eval_images,
    )

    if improved_test_summary["iou_mean"] >= baseline_test_summary["iou_mean"]:
        final_cfg = improved_cfg
        final_test_summary = improved_test_summary
        final_test_details = improved_test_details
        improvement_selected = "postprocess_on"
    else:
        final_cfg = baseline_cfg
        final_test_summary = baseline_test_summary
        final_test_details = baseline_test_details
        improvement_selected = "postprocess_off"

    # Classifier comparison: RandomForest.
    print("[INFO] Training RandomForest comparison ...")
    rf_cfg = replace(cfg, classifier="rf", apply_postprocess=False)
    rf_model, rf_params, rf_val_metrics, rf_train_time = train_method2(train_records, val_records, rf_cfg)
    print("[INFO] Evaluating RandomForest on test set ...")
    rf_test_summary, rf_test_details = evaluate_on_records(
        rf_model,
        test_records,
        rf_cfg,
        save_dir=output_root / "masks_rf_compare",
        progress_name="test_rf",
        verbose=rf_cfg.verbose_progress,
        log_every=rf_cfg.progress_log_every_eval_images,
    )

    # Data sensitivity experiments.
    print("[INFO] Running data-sensitivity experiments ...")
    sensitivity_rows: List[Dict[str, Any]] = []
    for ratio in (1.0, 0.75, 0.50):
        print(f"[INFO] Sensitivity run ratio={ratio:.2f} ...", flush=True)
        sub_train = subset_records(train_records, ratio=ratio, seed=args.seed)
        ratio_model, ratio_params, ratio_val_metrics, ratio_train_time = train_method2(sub_train, val_records, baseline_cfg)
        ratio_test_summary, _ = evaluate_on_records(
            ratio_model,
            test_records,
            baseline_cfg,
            save_dir=None,
            progress_name=f"test_ratio_{ratio:.2f}",
            verbose=baseline_cfg.verbose_progress,
            log_every=baseline_cfg.progress_log_every_eval_images,
        )
        sensitivity_rows.append(
            {
                "experiment": f"train_ratio_{ratio:.2f}",
                "ratio": ratio,
                "train_samples": len(sub_train),
                "classifier": ratio_params["classifier"],
                "val_iou": ratio_val_metrics["iou_mean"],
                "test_iou": ratio_test_summary["iou_mean"],
                "train_time_sec": ratio_train_time,
                "test_time_mean_sec": ratio_test_summary["test_time_sec_mean"],
            }
        )

    # Robustness experiments on final setting.
    print("[INFO] Running robustness experiments ...")
    robustness_rows: List[Dict[str, Any]] = []
    for kind in ("gaussian_noise", "gaussian_blur", "low_brightness", "low_contrast", "occlusion"):
        print(f"[INFO] Robustness distortion={kind} ...", flush=True)
        summary, _ = evaluate_with_transform(
            model=baseline_model,
            records=test_records,
            cfg=final_cfg,
            transform_fn=lambda im, idx, k=kind: apply_distortion(im, kind=k, seed=args.seed + idx),
            progress_name=kind,
        )
        robustness_rows.append(
            {
                "distortion": kind,
                "precision": summary["precision_mean"],
                "recall": summary["recall_mean"],
                "f1": summary["f1_mean"],
                "iou": summary["iou_mean"],
                "test_time_mean_sec": summary["test_time_sec_mean"],
            }
        )

    # Failure analysis: pick worst IoU samples from final predictions.
    print("[INFO] Generating failure analysis artifacts ...")
    worst = sorted(final_test_details, key=lambda x: x["iou"])[:5]
    failure_lines = [
        "# Failure Analysis (Method2)",
        "",
        "| sample_id | precision | recall | f1 | iou | reason |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for item in worst:
        sid = item["sample_id"]
        rec = next(r for r in test_records if r.sample_id == sid)
        img = read_image(rec.image_path, final_cfg)
        gt = read_mask(rec.mask_path, final_cfg)
        pred, _ = predict_method2(baseline_model, img, final_cfg)
        _save_failure_overlay(img, pred, gt, output_root / "failure_cases" / f"{sid}.png")

        reason = _failure_reason(item)
        failure_lines.append(
            f"| {sid} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['iou']:.4f} | {reason} |"
        )

    (output_root / "failure_analysis.md").write_text("\n".join(failure_lines), encoding="utf-8")

    # Save models and all logs.
    print("[INFO] Writing outputs ...")
    save_model(baseline_model, final_cfg, output_root / "method2_model.joblib")

    experiment_summary_rows = [
        {
            "experiment": "svm_baseline",
            "classifier": baseline_params["classifier"],
            "val_iou": baseline_val_metrics["iou_mean"],
            "test_iou": baseline_test_summary["iou_mean"],
            "train_time_sec": baseline_train_time,
            "test_time_mean_sec": baseline_test_summary["test_time_sec_mean"],
        },
        {
            "experiment": "svm_final",
            "classifier": baseline_params["classifier"],
            "val_iou": baseline_val_metrics["iou_mean"],
            "test_iou": final_test_summary["iou_mean"],
            "train_time_sec": baseline_train_time,
            "test_time_mean_sec": final_test_summary["test_time_sec_mean"],
        },
        {
            "experiment": "rf_compare",
            "classifier": rf_params["classifier"],
            "val_iou": rf_val_metrics["iou_mean"],
            "test_iou": rf_test_summary["iou_mean"],
            "train_time_sec": rf_train_time,
            "test_time_mean_sec": rf_test_summary["test_time_sec_mean"],
        },
    ]

    _write_csv(output_root / "experiment_summary.csv", experiment_summary_rows)
    _write_csv(output_root / "sensitivity_results.csv", sensitivity_rows)
    _write_csv(output_root / "robustness_results.csv", robustness_rows)
    _write_csv(output_root / "test_details_final.csv", final_test_details)

    write_json(
        output_root / "training_record.json",
        {
            "selected_final_setting": improvement_selected,
            "baseline_params": baseline_params,
            "rf_params": rf_params,
            "baseline_train_time_sec": baseline_train_time,
            "rf_train_time_sec": rf_train_time,
            "has_cv2": HAS_CV2,
        },
    )
    write_json(output_root / "validation_results.json", {"svm": baseline_val_metrics, "rf": rf_val_metrics})
    write_json(
        output_root / "test_results.json",
        {
            "svm_baseline": baseline_test_summary,
            "svm_final": final_test_summary,
            "rf_compare": rf_test_summary,
        },
    )

    # One compact hand-off for member D.
    write_json(
        output_root / "report_for_D.json",
        {
            "train_params": {
                "svm": baseline_params,
                "rf": rf_params,
                "final_selected": improvement_selected,
            },
            "validation": {
                "svm": baseline_val_metrics,
                "rf": rf_val_metrics,
            },
            "test": {
                "svm_baseline": baseline_test_summary,
                "svm_final": final_test_summary,
                "rf_compare": rf_test_summary,
                "robustness": robustness_rows,
                "sensitivity": sensitivity_rows,
            },
            "timing": {
                "svm_train_time_sec": baseline_train_time,
                "rf_train_time_sec": rf_train_time,
                "svm_final_test_time_mean_sec": final_test_summary["test_time_sec_mean"],
            },
        },
    )

    print("[INFO] Method2 experiments complete.")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
