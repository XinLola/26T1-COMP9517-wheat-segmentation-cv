from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def discover_pairs(test_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img_path in sorted(test_dir.glob('*.png')):
        if img_path.stem.endswith('_mask'):
            continue
        gt_path = test_dir / f'{img_path.stem}_mask.png'
        if gt_path.exists():
            pairs.append((img_path, gt_path))
    if not pairs:
        raise FileNotFoundError(f'No image/mask pairs found in {test_dir}')
    return pairs


def read_binary_mask(path: Path, plant_is_black: bool = True) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f'Cannot read {path}')
    if plant_is_black:
        return (arr == 0)
    return (arr > 0)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
    }


def summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for key in ['precision', 'recall', 'f1', 'iou', 'test_time_sec']:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[f'{key}_mean'] = float(values.mean())
        out[f'{key}_std'] = float(values.std())
    return out


def evaluate_method(method_name: str, pred_dir: Path, test_dir: Path) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    pairs = discover_pairs(test_dir)
    rows = []
    for img_path, gt_path in pairs:
        pred_path = pred_dir / img_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f'Missing prediction for {img_path.name} in {pred_dir}')

        t0 = time.perf_counter()
        pred = read_binary_mask(pred_path, plant_is_black=True)
        elapsed = time.perf_counter() - t0
        gt = read_binary_mask(gt_path, plant_is_black=True)
        m = compute_metrics(pred, gt)
        rows.append({
            'method': method_name,
            'image_name': img_path.name,
            **m,
            'test_time_sec': float(elapsed),
        })
    return summarise(rows), rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified evaluation for COMP9517 D role')
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--method', action='append', nargs=2, metavar=('NAME', 'PRED_DIR'), required=True,
                        help='Example: --method method1 output/method1/masks')
    parser.add_argument('--out-dir', type=Path, default=Path('output/D_eval'))
    args = parser.parse_args()

    all_summary = []
    for name, pred_dir_raw in args.method:
        pred_dir = Path(pred_dir_raw)
        summary, rows = evaluate_method(name, pred_dir, args.test_dir)
        all_summary.append({'method': name, **summary})
        write_csv(args.out_dir / f'{name}_per_image.csv', rows)

    write_csv(args.out_dir / 'summary.csv', all_summary)
    (args.out_dir / 'summary.json').write_text(json.dumps(all_summary, indent=2), encoding='utf-8')
    print(json.dumps(all_summary, indent=2))


if __name__ == '__main__':
    main()
