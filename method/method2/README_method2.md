# Method2 (HSV + LBP + Gabor + Classifier)

This folder implements `method2_ml.py` and the end-to-end experiment runner.

## Files

- `method2_ml.py`: core APIs
  - `extract_features(image, mask=None, cfg)`
  - `train_method2(train_set, val_set, cfg)`
  - `predict_method2(model, image, cfg)`
  - `save_mask(pred_mask, out_path)`
  - `evaluate(pred_mask, gt_mask)`
- `run_method2_experiments.py`: full experiment matrix
- `tests/test_method2_ml.py`: sanity checks

## Supported Dataset Layouts

Any one of the following:

1. `dataset_root/train/images`, `dataset_root/train/masks`, ...
2. `dataset_root/images/train`, `dataset_root/masks/train`, ...
3. `dataset_root/images`, `dataset_root/masks`, and `dataset_root/splits/train.txt|val.txt|test.txt`
4. `dataset_root/train`, `dataset_root/validation`, `dataset_root/test` (flat folders with image + `*_mask` files)

`val` and `validation` are both accepted.

## Install

```powershell
& "C:\Users\19087\Desktop\forGao\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Run

Default run (if `EWS-Dataset` is next to this script):

```powershell
& "C:\Users\19087\Desktop\forGao\.venv\Scripts\python.exe" "C:\Users\19087\Desktop\forGao\groupProjectSpecification\method2\run_method2_experiments.py"
```

Custom dataset path:

```powershell
& "C:\Users\19087\Desktop\forGao\.venv\Scripts\python.exe" "C:\Users\19087\Desktop\forGao\groupProjectSpecification\method2\run_method2_experiments.py" --dataset-root "D:\EWS-Dataset" --output-root "C:\Users\19087\Desktop\forGao\groupProjectSpecification\method2\outputs\method2"
```

Optional progress controls:

```powershell
# print progress every 5 images
& "C:\Users\19087\Desktop\forGao\.venv\Scripts\python.exe" "C:\Users\19087\Desktop\forGao\groupProjectSpecification\method2\run_method2_experiments.py" --log-every 5

# less logs
& "C:\Users\19087\Desktop\forGao\.venv\Scripts\python.exe" "C:\Users\19087\Desktop\forGao\groupProjectSpecification\method2\run_method2_experiments.py" --quiet
```

## Main Outputs

Under `outputs/method2/`:

- `masks_final/`
- `training_record.json`
- `validation_results.json`
- `test_results.json`
- `robustness_results.csv`
- `sensitivity_results.csv`
- `failure_analysis.md` + `failure_cases/*.png`
- `report_for_D.json`
