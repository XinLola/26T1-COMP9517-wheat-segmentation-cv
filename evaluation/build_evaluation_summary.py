from pathlib import Path
import json
import csv


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(d, keys, default=None):
    for k in keys:
        if d is None or k not in d:
            return default
        d = d[k]
    return d


def build_row(method_name, summary):
    test_block = safe_get(summary, ["test"])

    if "baseline" in test_block:
        base = test_block["baseline"]
    elif "svm_final" in test_block:
        base = test_block["svm_final"]
    else:
        base = {}

    return {
        "method": method_name,
        "precision": base.get("precision_mean"),
        "recall": base.get("recall_mean"),
        "f1": base.get("f1_mean"),
        "iou": base.get("iou_mean"),
        # std will be None because there is only one split (no averaging over multiple splits)
        "precision_std": base.get("precision_std", None),
        "recall_std": base.get("recall_std", None),
        "f1_std": base.get("f1_std", None),
        "iou_std": base.get("iou_std", None),

        "avg_inference_time_sec": safe_get(summary, ["timing", "test_time_mean_sec"]),
    }


def read_method(output_root, method_name):
    path = output_root / method_name / "evaluation_summary.json"
    summary = load_json(path)
    if not summary:
        print(f"[WARN] Missing {path}")
        return None

    row = build_row(method_name, summary)

    if all(v is None for v in row.values()):
        return None

    return row

def write_markdown(rows, path):
    with path.open("w", encoding="utf-8") as f:
        f.write("## 📊 Quantitative Comparison\n\n")
        f.write("| Method | Precision | Recall | F1-score | IoU | Inference Time (s) |\n")
        f.write("|--------|----------|--------|----------|-----|--------------------|\n")

        for r in rows:
            def fmt(mean, std):
                if std is None:
                    return f"{mean:.3f}"
                return f"{mean:.3f} ± {std:.3f}"

            f.write(
                f"| {r['method']} | "
                f"{fmt(r['precision'], r['precision_std'])} | "
                f"{fmt(r['recall'], r['recall_std'])} | "
                f"{fmt(r['f1'], r['f1_std'])} | "
                f"{fmt(r['iou'], r['iou_std'])} | "
                f"{r['avg_inference_time_sec']:.3f} |\n"
            )

def main():
    root = Path(__file__).resolve().parent.parent
    output_root = root / "output"
    eval_root = root / "evaluation"
    eval_root.mkdir(exist_ok=True)

    method_names = ["method1", "method2", "method3", "method4"]

    rows = []
    for m in method_names:
        row = read_method(output_root, m)
        if row:
            rows.append(row)

    csv_path = eval_root / "quantitative_comparison.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "precision",
                    "recall",
                    "f1",
                    "iou",
                    "precision_std",
                    "recall_std",
                    "f1_std",
                    "iou_std",
                    "avg_inference_time_sec",
                ]
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[INFO] Saved comparison to {csv_path}")
    else:
        print("[ERROR] No valid evaluation_summary found.")

    md_path = eval_root / "quantitative_comparison.md"
    write_markdown(rows, md_path)
    print(f"[INFO] Saved markdown to {md_path}")


if __name__ == "__main__":
    main()