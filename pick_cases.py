import csv
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python pick_cases.py <per_image_csv>")
        return

    csv_path = Path(sys.argv[1])
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))

    for r in rows:
        r["iou"] = float(r["iou"])
        r["f1"] = float(r["f1"])

    rows_sorted = sorted(rows, key=lambda x: x["iou"])

    print(f"\n=== {csv_path.name} ===")

    print("\nWorst 3:")
    for r in rows_sorted[:3]:
        print(f'{r["image_name"]} | IoU={r["iou"]:.4f} | F1={r["f1"]:.4f}')

    print("\nBest 3:")
    for r in rows_sorted[-3:]:
        print(f'{r["image_name"]} | IoU={r["iou"]:.4f} | F1={r["f1"]:.4f}')

if __name__ == "__main__":
    main()