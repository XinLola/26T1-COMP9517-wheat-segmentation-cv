from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def add_gaussian_noise(img: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def apply_blur(img: np.ndarray, k: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def low_brightness(img: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    mean = img.mean(axis=(0, 1), keepdims=True)
    out = (img.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def occlusion(img: np.ndarray, frac: float = 0.18) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    oh, ow = int(h * frac), int(w * frac)
    y0 = h // 2 - oh // 2
    x0 = w // 2 - ow // 2
    out[y0:y0+oh, x0:x0+ow] = 0
    return out


DISTORTIONS = {
    'gaussian_noise': add_gaussian_noise,
    'gaussian_blur': apply_blur,
    'low_brightness': low_brightness,
    'low_contrast': low_contrast,
    'occlusion': occlusion,
}


def discover_test_images(test_dir: Path):
    for img_path in sorted(test_dir.glob('*.png')):
        if not img_path.stem.endswith('_mask'):
            yield img_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Create distorted test set for robustness analysis')
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('output/distorted_test'))
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    for name, fn in DISTORTIONS.items():
        ddir = args.out_dir / name
        ddir.mkdir(parents=True, exist_ok=True)
        for img_path in discover_test_images(args.test_dir):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out = fn(img)
            cv2.imwrite(str(ddir / img_path.name), out)
            rows.append({'distortion': name, 'image_name': img_path.name, 'saved_to': str(ddir / img_path.name)})

    csv_path = args.out_dir / 'distortion_index.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['distortion', 'image_name', 'saved_to'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved to {args.out_dir}')


if __name__ == '__main__':
    main()
