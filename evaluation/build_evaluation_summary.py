from pathlib import Path
import json
import csv


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_get(d, keys, default=None):
    for k in keys:
        if d is None or k not in d:
            return default
        d = d[k]
    return d


def mean_from_csv(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def has_valid_metrics(row):
    return any(
        row.get(k) is not None
        for k in ["precision", "recall", "f1", "iou", "avg_inference_time_sec"]
    )


def build_row(method_name, precision, recall, f1, iou, avg_inference_time_sec):
    return {
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "avg_inference_time_sec": avg_inference_time_sec,
    }


def read_method1(output_root):
    metrics = load_json(output_root / "method1" / "metrics.json")
    if metrics:
        row = build_row(
            "Method 1",
            safe_get(metrics, ["precision"]),
            safe_get(metrics, ["recall"]),
            safe_get(metrics, ["f1"]),
            safe_get(metrics, ["iou"]),
            safe_get(metrics, ["avg_time"]),
        )
        if has_valid_metrics(row):
            return row

    report = load_json(output_root / "method1" / "report_for_D.json")
    if report:
        row = build_row(
            "Method 1",
            safe_get(report, ["test", "precision"]),
            safe_get(report, ["test", "recall"]),
            safe_get(report, ["test", "f1"]),
            safe_get(report, ["test", "iou"]),
            safe_get(report, ["test", "avg_inference_time_sec"]),
        )
        if has_valid_metrics(row):
            return row

    rows = load_csv_rows(output_root / "method1" / "test_details_final.csv")
    if rows:
        row = build_row(
            "Method 1",
            mean_from_csv(rows, "precision"),
            mean_from_csv(rows, "recall"),
            mean_from_csv(rows, "f1"),
            mean_from_csv(rows, "iou"),
            mean_from_csv(rows, "test_time_sec"),
        )
        if has_valid_metrics(row):
            return row

    return None


def read_method2(output_root):
    report = load_json(output_root / "method2" / "test_results.json")
    if report:
        row = build_row(
            "Method 2",
            safe_get(report, ["svm_final", "precision_mean"]),
            safe_get(report, ["svm_final", "recall_mean"]),
            safe_get(report, ["svm_final", "f1_mean"]),
            safe_get(report, ["svm_final", "iou_mean"]),
            safe_get(report, ["svm_final", "test_time_sec_mean"]),
        )
        if has_valid_metrics(row):
            return row

    rows = load_csv_rows(output_root / "method2" / "test_details_final.csv")
    if rows:
        row = build_row(
            "Method 2",
            mean_from_csv(rows, "precision"),
            mean_from_csv(rows, "recall"),
            mean_from_csv(rows, "f1"),
            mean_from_csv(rows, "iou"),
            mean_from_csv(rows, "test_time_sec"),
        )
        if has_valid_metrics(row):
            return row

    return None


def read_method3(output_root):
    metrics = load_json(output_root / "method3" / "metrics.json")
    if metrics:
        row = build_row(
            "Method 3",
            safe_get(metrics, ["precision"]),
            safe_get(metrics, ["recall"]),
            safe_get(metrics, ["f1"]),
            safe_get(metrics, ["iou"]),
            safe_get(metrics, ["avg_time"]),
        )
        if has_valid_metrics(row):
            return row

    report = load_json(output_root / "method3" / "report_for_D.json")
    if report:
        row = build_row(
            "Method 3",
            safe_get(report, ["test", "precision"]),
            safe_get(report, ["test", "recall"]),
            safe_get(report, ["test", "f1"]),
            safe_get(report, ["test", "iou"]),
            safe_get(report, ["test", "avg_inference_time_sec"]),
        )
        if has_valid_metrics(row):
            return row

    rows = load_csv_rows(output_root / "method3" / "test_details_final.csv")
    if rows:
        row = build_row(
            "Method 3",
            mean_from_csv(rows, "precision"),
            mean_from_csv(rows, "recall"),
            mean_from_csv(rows, "f1"),
            mean_from_csv(rows, "iou"),
            mean_from_csv(rows, "test_time_sec"),
        )
        if has_valid_metrics(row):
            return row

    return None


def read_method4(output_root):
    metrics = load_json(output_root / "method4" / "metrics.json")
    if metrics:
        row = build_row(
            "Method 4",
            safe_get(metrics, ["metrics", "precision"]),
            safe_get(metrics, ["metrics", "recall"]),
            safe_get(metrics, ["metrics", "f1"]),
            safe_get(metrics, ["metrics", "iou"]),
            safe_get(metrics, ["metrics", "average_predict_time_per_image"]),
        )
        if has_valid_metrics(row):
            return row
    return None


def main():
    root = Path(__file__).resolve().parent.parent
    output_root = root / "output"
    eval_root = root / "evaluation"
    eval_root.mkdir(exist_ok=True)

    readers = [
        read_method1,
        read_method2,
        read_method3,
        read_method4,
    ]

    rows = []
    for reader in readers:
        row = reader(output_root)
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
                    "avg_inference_time_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved comparison to {csv_path}")
    else:
        print("No data found. Check output folders.")


if __name__ == "__main__":
    main()