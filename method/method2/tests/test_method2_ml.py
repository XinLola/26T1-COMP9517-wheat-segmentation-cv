import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parents[1]))

from method2_ml import Method2Config, evaluate, extract_features, predict_method2


def _synthetic_image_and_mask(h: int = 32, w: int = 32):
    img = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    center = ((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) < (min(h, w) / 3.5) ** 2
    img[..., 1] = np.where(center, 0.9, 0.2)
    img[..., 0] = 0.2
    img[..., 2] = 0.2
    mask = center.astype(np.uint8)
    return img, mask


def test_extract_features_shape_and_finite():
    cfg = Method2Config()
    img, _ = _synthetic_image_and_mask()
    x = extract_features(img, cfg=cfg)
    assert x.ndim == 2
    assert x.shape[0] == img.shape[0] * img.shape[1]
    assert np.isfinite(x).all()


def test_predict_method2_shape_and_binary():
    cfg = Method2Config(apply_postprocess=False)
    img, mask = _synthetic_image_and_mask()
    x = extract_features(img, cfg=cfg)
    y = mask.reshape(-1)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
        ]
    )
    model.fit(x, y)

    pred, elapsed = predict_method2(model, img, cfg)
    assert pred.shape == mask.shape
    assert elapsed >= 0.0
    unique_vals = set(np.unique(pred).tolist())
    assert unique_vals.issubset({0, 1})


def test_evaluate_metric_range():
    _, mask = _synthetic_image_and_mask()
    metric = evaluate(mask, mask)
    for key in ("precision", "recall", "f1", "iou"):
        assert 0.0 <= metric[key] <= 1.0
    assert metric["iou"] == 1.0


if __name__ == "__main__":
    try:
        import pytest

        raise SystemExit(pytest.main([__file__, "-q"]))
    except ModuleNotFoundError:
        # Fallback: allow direct execution without pytest installed.
        test_extract_features_shape_and_finite()
        test_predict_method2_shape_and_binary()
        test_evaluate_metric_range()
        print("3 direct checks passed (pytest not installed).")
