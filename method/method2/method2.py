"""
Method 2 - Wheat Segmentation ML Pipeline
=========================================
Usage:
  python method2.py
  python method2.py --dataset-root /path/to/data
  python method2.py --output-root output/method2
  python method2.py --image-size 256x256
  python method2.py --seed 42
  python method2.py --log-every 10
  python method2.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from skimage import color, exposure, filters, io, morphology, transform, util
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass
class Method2Config:
    image_size: Optional[Tuple[int, int]] = None
    normalize_input: bool = True
    mask_threshold: float = 0.5

    lbp_radius: int = 1
    lbp_points: int = 8
    gabor_frequencies: Tuple[float, ...] = (0.10, 0.20)
    gabor_thetas: Tuple[float, ...] = (
        0.0,
        float(np.pi / 4.0),
        float(np.pi / 2.0),
        float(3.0 * np.pi / 4.0),
    )
    use_xy_feature: bool = True

    classifier: str = "svm"  # svm | rf
    # Fast default: keep one strong baseline setting first.
    # You can expand this grid for final report runs.
    svm_c_values: Tuple[float, ...] = (1.0,)
    svm_gamma_values: Tuple[Any, ...] = ("scale",)
    rf_n_estimators: Tuple[int, ...] = (200, 400)
    rf_max_depth: Tuple[Optional[int], ...] = (None, 20)
    rf_n_jobs: int = 1

    # Fast default for SVM training cost control.
    max_train_pixels_per_image: int = 1500
    max_val_pixels_per_image: int = 4000
    max_train_pixels_total: int = 40000
    max_val_pixels_total: int = 80000

    apply_postprocess: bool = True
    post_open_radius: int = 1
    post_close_radius: int = 2

    random_state: int = 42
    output_foreground_value: int = 255
    verbose_progress: bool = True
    progress_log_every_images: int = 10
    progress_log_every_eval_images: int = 10


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _clean_stem(name: str) -> str:
    n = name.lower().strip()
    for suffix in ("_mask", "-mask", " mask"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


def _is_mask_stem(name: str) -> bool:
    n = name.lower().strip()
    return n.endswith("_mask") or n.endswith("-mask") or n.endswith(" mask")


def _find_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file()])


def _pair_images_masks(images_dir: Path, masks_dir: Path) -> List[SampleRecord]:
    image_files = _find_image_files(images_dir)
    mask_files = _find_image_files(masks_dir)

    if not image_files:
        raise FileNotFoundError(f"No image files found in: {images_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in: {masks_dir}")

    mask_by_stem = {_clean_stem(m.stem): m for m in mask_files}
    records: List[SampleRecord] = []

    for img in image_files:
        key = _clean_stem(img.stem)
        if key not in mask_by_stem:
            continue
        records.append(SampleRecord(sample_id=img.stem, image_path=img, mask_path=mask_by_stem[key]))

    if not records:
        raise ValueError(f"No paired image/mask found under {images_dir} and {masks_dir}")
    return records


def _pair_flat_split_dir(split_dir: Path) -> List[SampleRecord]:
    files = _find_image_files(split_dir)
    if not files:
        raise FileNotFoundError(f"No image files found in: {split_dir}")

    image_by_key: Dict[str, Path] = {}
    mask_by_key: Dict[str, Path] = {}

    for p in files:
        stem = p.stem
        key = _clean_stem(stem)
        if _is_mask_stem(stem):
            mask_by_key[key] = p
        else:
            image_by_key[key] = p

    records: List[SampleRecord] = []
    for key, image_path in image_by_key.items():
        mask_path = mask_by_key.get(key)
        if mask_path is None:
            continue
        records.append(SampleRecord(sample_id=image_path.stem, image_path=image_path, mask_path=mask_path))

    if not records:
        raise ValueError(f"No paired image/mask found in flat split dir: {split_dir}")
    return sorted(records, key=lambda r: r.sample_id)


def discover_split_records(dataset_root: Path, split: str) -> List[SampleRecord]:
    split = split.lower().strip()

    split_aliases = [split]
    if split == "val":
        split_aliases.append("validation")
    elif split == "validation":
        split_aliases.append("val")

    for sp in split_aliases:
        candidates = [
            (dataset_root / sp / "images", dataset_root / sp / "masks"),
            (dataset_root / "images" / sp, dataset_root / "masks" / sp),
        ]

        for images_dir, masks_dir in candidates:
            if images_dir.exists() and masks_dir.exists():
                return _pair_images_masks(images_dir, masks_dir)

        flat_split_dir = dataset_root / sp
        if flat_split_dir.is_dir():
            try:
                return _pair_flat_split_dir(flat_split_dir)
            except (FileNotFoundError, ValueError):
                pass

    split_file_candidates = [
        dataset_root / f"{split}.txt",
        dataset_root / "splits" / f"{split}.txt",
    ]
    for split_file in split_file_candidates:
        if not split_file.exists():
            continue

        images_dir = dataset_root / "images"
        masks_dir = dataset_root / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"Found split file {split_file} but missing images/ and masks/ under {dataset_root}"
            )

        entries = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        mask_index = {_clean_stem(p.stem): p for p in _find_image_files(masks_dir)}
        records: List[SampleRecord] = []
        for item in entries:
            item_stem = _clean_stem(Path(item).stem)
            image_path = images_dir / item
            if not image_path.exists():
                for ext in _IMAGE_SUFFIXES:
                    c = images_dir / f"{Path(item).stem}{ext}"
                    if c.exists():
                        image_path = c
                        break
            if not image_path.exists() or item_stem not in mask_index:
                continue
            records.append(SampleRecord(sample_id=image_path.stem, image_path=image_path, mask_path=mask_index[item_stem]))

        if not records:
            raise ValueError(f"No valid records resolved from split file: {split_file}")
        return records

    raise FileNotFoundError(
        f"Unable to locate split={split}. Supported layouts:\n"
        "1) <root>/<split>/images + <root>/<split>/masks\n"
        "2) <root>/images/<split> + <root>/masks/<split>\n"
        "3) <root>/images + <root>/masks + <root>/splits/<split>.txt\n"
        "4) <root>/<split> (flat folder with image + *_mask files)"
    )


def _resize_image(image: np.ndarray, image_size: Tuple[int, int], is_mask: bool) -> np.ndarray:
    target_h, target_w = image_size
    if HAS_CV2:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return resized

    order = 0 if is_mask else 1
    anti_aliasing = not is_mask
    resized = transform.resize(
        image,
        (target_h, target_w) if image.ndim == 2 else (target_h, target_w, image.shape[2]),
        order=order,
        preserve_range=True,
        anti_aliasing=anti_aliasing,
    )
    return resized


def read_image(path: Path, cfg: Method2Config) -> np.ndarray:
    if HAS_CV2:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = rgb.astype(np.float32)
        if cfg.normalize_input:
            image /= 255.0
    else:
        arr = io.imread(str(path))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        image = util.img_as_float32(arr) if cfg.normalize_input else arr.astype(np.float32)

    if cfg.image_size is not None:
        image = _resize_image(image, cfg.image_size, is_mask=False)

    if cfg.normalize_input and image.max() > 1.0:
        image = image / 255.0

    return np.clip(image.astype(np.float32), 0.0, 1.0 if cfg.normalize_input else image.max())


def read_mask(path: Path, cfg: Method2Config) -> np.ndarray:
    if HAS_CV2:
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError(f"Failed to read mask: {path}")
        mask = raw.astype(np.float32)
    else:
        raw = io.imread(str(path))
        if raw.ndim == 3:
            raw = color.rgb2gray(raw)
        mask = raw.astype(np.float32)

    if cfg.image_size is not None:
        mask = _resize_image(mask, cfg.image_size, is_mask=True)

    threshold = cfg.mask_threshold * (255.0 if mask.max() > 1.0 else 1.0)
    # EWS dataset convention: black (0) = plant (foreground), white (255) = background
    # So plant pixels have mask value < threshold → label 1 (positive / foreground)
    binary = (mask <= threshold).astype(np.uint8)
    return binary


def extract_features(image: np.ndarray, mask: Optional[np.ndarray] = None, cfg: Optional[Method2Config] = None) -> np.ndarray:
    del mask  # Kept for interface compatibility.

    local_cfg = cfg or Method2Config()
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    h, w, c = img.shape
    if c != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {img.shape}")

    hsv = color.rgb2hsv(img).astype(np.float32)
    gray = color.rgb2gray(img).astype(np.float32)
    gray_u8 = util.img_as_ubyte(gray)

    lbp = local_binary_pattern(gray_u8, P=local_cfg.lbp_points, R=local_cfg.lbp_radius, method="uniform").astype(np.float32)
    lbp /= max(float(lbp.max()), 1e-6)

    components: List[np.ndarray] = [
        img[:, :, 0],
        img[:, :, 1],
        img[:, :, 2],
        hsv[:, :, 0],
        hsv[:, :, 1],
        hsv[:, :, 2],
        lbp,
    ]

    for freq in local_cfg.gabor_frequencies:
        for theta in local_cfg.gabor_thetas:
            real, imag = filters.gabor(gray, frequency=float(freq), theta=float(theta))
            magnitude = np.sqrt(real * real + imag * imag).astype(np.float32)
            components.append(magnitude)

    if local_cfg.use_xy_feature:
        yy, xx = np.mgrid[0:h, 0:w]
        x_norm = xx.astype(np.float32) / max(w - 1, 1)
        y_norm = yy.astype(np.float32) / max(h - 1, 1)
        components.append(x_norm)
        components.append(y_norm)

    feature_tensor = np.stack(components, axis=-1).astype(np.float32)
    return feature_tensor.reshape(h * w, feature_tensor.shape[-1])


def _balanced_pixel_sample(X: np.ndarray, y: np.ndarray, max_pixels: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if max_pixels <= 0 or len(y) <= max_pixels:
        return X, y

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        chosen = rng.choice(len(y), size=max_pixels, replace=False)
        return X[chosen], y[chosen]

    target_half = max_pixels // 2
    n_pos = min(len(pos_idx), target_half)
    n_neg = min(len(neg_idx), target_half)

    selected_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    selected_neg = rng.choice(neg_idx, size=n_neg, replace=False)

    selected = np.concatenate([selected_pos, selected_neg])
    if len(selected) < max_pixels:
        remaining = np.setdiff1d(np.arange(len(y)), selected, assume_unique=False)
        extra = min(max_pixels - len(selected), len(remaining))
        if extra > 0:
            selected = np.concatenate([selected, rng.choice(remaining, size=extra, replace=False)])

    rng.shuffle(selected)
    return X[selected], y[selected]


def _collect_split_pixels(
    records: Sequence[SampleRecord],
    cfg: Method2Config,
    max_pixels_per_image: int,
    max_pixels_total: int,
    rng: np.random.Generator,
    split_name: str = "split",
    verbose: bool = False,
    log_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    n_total = len(records)
    t0 = time.perf_counter()

    _log(
        verbose,
        f"[PROGRESS][PIXEL][{split_name}] start | samples={n_total}, max_pixels_per_image={max_pixels_per_image}, max_pixels_total={max_pixels_total}",
    )

    for i, rec in enumerate(records, start=1):
        image = read_image(rec.image_path, cfg)
        gt = read_mask(rec.mask_path, cfg)
        features = extract_features(image=image, mask=gt, cfg=cfg)
        labels = gt.reshape(-1).astype(np.uint8)
        xs, ys = _balanced_pixel_sample(features, labels, max_pixels_per_image, rng)
        x_parts.append(xs)
        y_parts.append(ys)
        if verbose and (i == 1 or i % max(1, log_every) == 0 or i == n_total):
            elapsed = time.perf_counter() - t0
            pct = 100.0 * i / max(n_total, 1)
            _log(
                True,
                f"[PROGRESS][PIXEL][{split_name}] {i}/{n_total} ({pct:.1f}%) | sample={rec.sample_id} | sampled_pixels={len(ys)} | elapsed={elapsed:.1f}s",
            )

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if len(y) > max_pixels_total:
        pick = rng.choice(len(y), size=max_pixels_total, replace=False)
        X = X[pick]
        y = y[pick]

    _log(
        verbose,
        f"[PROGRESS][PIXEL][{split_name}] done | final_pixels={len(y)} | feature_dim={X.shape[1]} | elapsed={time.perf_counter() - t0:.1f}s",
    )

    return X.astype(np.float32), y.astype(np.uint8)


def _build_candidate_models(cfg: Method2Config) -> List[Tuple[Pipeline, Dict[str, Any]]]:
    out: List[Tuple[Pipeline, Dict[str, Any]]] = []

    if cfg.classifier.lower() == "svm":
        for c in cfg.svm_c_values:
            for gamma in cfg.svm_gamma_values:
                pipe = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            SVC(
                                kernel="rbf",
                                C=float(c),
                                gamma=gamma,
                                class_weight="balanced",
                                cache_size=512,
                            ),
                        ),
                    ]
                )
                out.append((pipe, {"classifier": "svm", "C": float(c), "gamma": gamma}))

    elif cfg.classifier.lower() == "rf":
        for n_estimators in cfg.rf_n_estimators:
            for max_depth in cfg.rf_max_depth:
                pipe = Pipeline(
                    steps=[
                        ("scaler", "passthrough"),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=int(n_estimators),
                                max_depth=max_depth,
                                class_weight="balanced_subsample",
                                n_jobs=cfg.rf_n_jobs,
                                random_state=cfg.random_state,
                            ),
                        ),
                    ]
                )
                out.append(
                    (
                        pipe,
                        {
                            "classifier": "rf",
                            "n_estimators": int(n_estimators),
                            "max_depth": max_depth,
                        },
                    )
                )
    else:
        raise ValueError(f"Unsupported classifier={cfg.classifier}. Use svm or rf.")

    return out


def _binary_postprocess(mask: np.ndarray, cfg: Method2Config) -> np.ndarray:
    out = (mask > 0).astype(bool)

    if cfg.post_open_radius > 0:
        out = morphology.binary_opening(out, morphology.disk(cfg.post_open_radius))
    if cfg.post_close_radius > 0:
        out = morphology.binary_closing(out, morphology.disk(cfg.post_close_radius))

    return out.astype(np.uint8)


def predict_method2(model: Pipeline, image: np.ndarray, cfg: Method2Config) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    h, w, _ = image.shape

    x = extract_features(image=image, mask=None, cfg=cfg)
    pred = model.predict(x).astype(np.uint8).reshape(h, w)

    if cfg.apply_postprocess:
        pred = _binary_postprocess(pred, cfg)

    elapsed = time.perf_counter() - start
    return pred.astype(np.uint8), elapsed



def visuals(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Produce a side-by-side comparison image (Original | Prediction | Ground Truth).

    `image` can be either:
      - RGB float32 in [0, 1]  (returned by read_image with normalize_input=True)
      - BGR uint8 in [0, 255]  (used by methods 1/3/4)
    Both cases are handled automatically.
    """
    import cv2

    img = np.array(image)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if image.dtype == np.uint8:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Prediction: internal convention is 1 = plant, display plant as black
    pred_display = np.where(np.array(pred_mask) > 0, 0, 255).astype(np.uint8)
    pred_rgb = cv2.cvtColor(pred_display, cv2.COLOR_GRAY2RGB)

    # Ground truth: support both binary {0,1} and raw EWS {0,255}
    gt = np.array(gt_mask)
    if gt.max() <= 1.0 and gt.min() >= 0.0:
        gt_display = np.where(gt > 0, 0, 255).astype(np.uint8)
    else:
        gt_display = np.where(gt == 0, 0, 255).astype(np.uint8)

    gt_rgb = cv2.cvtColor(gt_display, cv2.COLOR_GRAY2RGB)

    panels = [img_rgb, pred_rgb, gt_rgb]
    titles = ["Original image", "Prediction mask", "Actual mask"]
    h, w = img_rgb.shape[:2]
    combined = []
    for panel, title in zip(panels, titles):
        canvas = np.full((h + 40, w, 3), 255, dtype=np.uint8)
        canvas[40:, :, :] = panel
        cv2.putText(canvas, title, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        combined.append(canvas)

    result = np.hstack(combined)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    
def evaluate(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = (pred_mask > 0)
    gt = (gt_mask > 0)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    union = int(np.logical_or(pred, gt).sum())
    iou = tp / union if union > 0 else 1.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def save_mask(pred_mask: np.ndarray, out_path: Path, cfg: Optional[Method2Config] = None) -> None:
    out_cfg = cfg or Method2Config()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    binary = (pred_mask > 0).astype(np.uint8)
    # Save using EWS dataset convention: 0 (black) = plant, 255 (white) = background
    # This is the inverse of the internal binary representation (1=plant)
    encoded = np.where(binary > 0, 0, 255).astype(np.uint8)

    if HAS_CV2:
        cv2.imwrite(str(out_path), encoded)
    else:
        io.imsave(str(out_path), encoded, check_contrast=False)


def evaluate_on_records(
    model: Pipeline,
    records: Sequence[SampleRecord],
    cfg: Method2Config,
    save_dir: Optional[Path] = None,
    progress_name: str = "eval",
    verbose: bool = False,
    log_every: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not records:
        raise ValueError("records is empty")

    details: List[Dict[str, Any]] = []
    n_total = len(records)
    t0 = time.perf_counter()
    _log(verbose, f"[PROGRESS][EVAL][{progress_name}] start | samples={n_total}")

    for i, rec in enumerate(records, start=1):
        image = read_image(rec.image_path, cfg)
        gt = read_mask(rec.mask_path, cfg)
        pred, test_time = predict_method2(model, image, cfg)
        m = evaluate(pred, gt)

        if save_dir is not None:
            save_mask(pred, save_dir / f"{rec.sample_id}.png", cfg)
            # also save visuals
            vis_dir = save_dir.parent / "visuals"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis = visuals(image, pred, gt)
            cv2.imwrite(str(vis_dir / f"{rec.sample_id}.png"), vis)

        details.append(
            {
                "sample_id": rec.sample_id,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "iou": m["iou"],
                "test_time_sec": float(test_time),
            }
        )

        if verbose and log_every > 0 and (i == 1 or i % max(1, log_every) == 0 or i == n_total):
            elapsed = time.perf_counter() - t0
            pct = 100.0 * i / max(n_total, 1)
            _log(
                True,
                f"[PROGRESS][EVAL][{progress_name}] {i}/{n_total} ({pct:.1f}%) | sample={rec.sample_id} | iou={m['iou']:.4f} | elapsed={elapsed:.1f}s",
            )

    summary: Dict[str, float] = {}
    for key in ("precision", "recall", "f1", "iou", "test_time_sec"):
        values = np.asarray([d[key] for d in details], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    _log(
        verbose,
        f"[PROGRESS][EVAL][{progress_name}] done | iou_mean={summary['iou_mean']:.4f}, f1_mean={summary['f1_mean']:.4f} | elapsed={time.perf_counter() - t0:.1f}s",
    )

    return summary, details


def train_method2(
    train_set: Sequence[SampleRecord],
    val_set: Sequence[SampleRecord],
    cfg: Method2Config,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float], float]:
    if not train_set:
        raise ValueError("train_set is empty")
    if not val_set:
        raise ValueError("val_set is empty")

    rng = np.random.default_rng(cfg.random_state)

    X_train, y_train = _collect_split_pixels(
        records=train_set,
        cfg=cfg,
        max_pixels_per_image=cfg.max_train_pixels_per_image,
        max_pixels_total=cfg.max_train_pixels_total,
        rng=rng,
        split_name="train",
        verbose=cfg.verbose_progress,
        log_every=cfg.progress_log_every_images,
    )

    candidates = _build_candidate_models(cfg)
    _log(
        cfg.verbose_progress,
        f"[PROGRESS][TRAIN] start grid search | classifier={cfg.classifier} | candidates={len(candidates)} | train_pixels={len(y_train)}",
    )

    best_model: Optional[Pipeline] = None
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, float] = {}
    best_train_time = 0.0
    best_iou = -1.0

    for cand_idx, (model, params) in enumerate(candidates, start=1):
        _log(
            cfg.verbose_progress,
            f"[PROGRESS][TRAIN] candidate {cand_idx}/{len(candidates)} start | params={params}",
        )
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_elapsed = time.perf_counter() - train_start

        val_metrics, _ = evaluate_on_records(
            model,
            val_set,
            cfg,
            save_dir=None,
            progress_name=f"val(cand={cand_idx})",
            verbose=cfg.verbose_progress,
            log_every=cfg.progress_log_every_eval_images,
        )
        val_iou = val_metrics["iou_mean"]
        _log(
            cfg.verbose_progress,
            f"[PROGRESS][TRAIN] candidate {cand_idx}/{len(candidates)} done | train_time={train_elapsed:.2f}s | val_iou={val_iou:.4f}",
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_model = model
            best_params = params
            best_metrics = val_metrics
            best_train_time = train_elapsed
            _log(
                cfg.verbose_progress,
                f"[PROGRESS][TRAIN] new best | candidate={cand_idx} | best_val_iou={best_iou:.4f}",
            )

    if best_model is None:
        raise RuntimeError("No candidate model trained.")

    _log(
        cfg.verbose_progress,
        f"[PROGRESS][TRAIN] finished | best_params={best_params} | best_val_iou={best_iou:.4f} | best_train_time={best_train_time:.2f}s",
    )

    return best_model, best_params, best_metrics, float(best_train_time)


def save_model(model: Pipeline, cfg: Method2Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "config": asdict(cfg),
    }
    joblib.dump(payload, path)


def load_model(path: Path) -> Tuple[Pipeline, Method2Config]:
    payload = joblib.load(path)
    model = payload["model"]
    config_dict = payload["config"]
    cfg = Method2Config(**config_dict)
    return model, cfg


def subset_records(records: Sequence[SampleRecord], ratio: float, seed: int) -> List[SampleRecord]:
    if not (0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")
    if ratio == 1.0:
        return list(records)

    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(records) * ratio)))
    idx = rng.choice(len(records), size=n, replace=False)
    idx = np.sort(idx)
    return [records[i] for i in idx]


def apply_distortion(image: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    k = kind.lower()
    if k == "gaussian_noise":
        noise = rng.normal(loc=0.0, scale=0.05, size=img.shape).astype(np.float32)
        out = np.clip(img + noise, 0.0, 1.0)
    elif k == "gaussian_blur":
        out = filters.gaussian(img, sigma=1.2, channel_axis=-1, preserve_range=True)
        out = np.clip(out, 0.0, 1.0)
    elif k == "low_brightness":
        out = np.clip(img * 0.60, 0.0, 1.0)
    elif k == "low_contrast":
        out = exposure.adjust_gamma(img, gamma=0.8, gain=0.8)
        out = np.clip(out, 0.0, 1.0)
    elif k == "occlusion":
        out = img.copy()
        h, w, _ = out.shape
        occ_h = max(1, int(h * 0.20))
        occ_w = max(1, int(w * 0.20))
        y0 = int(rng.integers(0, max(h - occ_h + 1, 1)))
        x0 = int(rng.integers(0, max(w - occ_w + 1, 1)))
        out[y0 : y0 + occ_h, x0 : x0 + occ_w, :] = 0.0
    else:
        raise ValueError(
            f"Unsupported distortion={kind}. Use gaussian_noise|gaussian_blur|low_brightness|low_contrast|occlusion"
        )

    return out.astype(np.float32)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        return "Under-segmentation: Possible broken boundaries or missing detection in low-contrast areas."
    if p < 0.55 and r >= 0.70:
        return "Over-segmentation: Possible misclassification of soil texture as crop."
    if iou < 0.40 and abs(p - r) < 0.15:
        return "Total confusion: Likely affected by lighting changes or severe noise."
    return "Complex failure: Insufficient feature separability and high background complexity."


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
    project_root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(description="Run Method2 experiments (HSV+LBP+Gabor + classifier).")
    p.add_argument("--dataset-root", type=Path, default=None, help="Dataset root path.")
    p.add_argument("--output-root", type=Path, default=project_root / "output" / "method2", help="Output folder.")
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
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate = project_root / "data"
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
    val_records = discover_split_records(dataset_root, "validation")
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
    write_json(output_root / "evaluation_summary.json", {
        "method": "Method 2 (ML: SVM + Features)",
        "type": "ml",

        "train": {
            "train_time_sec": baseline_train_time,
            "params": baseline_params
        },

        "validation": baseline_val_metrics,

        "test": {
            "baseline": final_test_summary,
            "robustness": robustness_rows,
            "extra": {
                "rf_compare": rf_test_summary,
                "sensitivity": sensitivity_rows
            }
        },

        "timing": {
            "train_time_sec": baseline_train_time,
            "test_time_mean_sec": final_test_summary["test_time_sec_mean"]
        }
    })

    print("[INFO] Method2 experiments complete.")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
