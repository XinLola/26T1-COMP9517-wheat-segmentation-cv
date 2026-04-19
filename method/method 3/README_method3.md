# Method 3: Random Forest with Handcrafted Features

## Overview
Method 3 implements a traditional machine learning pipeline using a Random Forest classifier. It extracts a 13-dimensional handcrafted feature vector per pixel (incorporating colour spaces, vegetation indices, local statistics, and texture) and trains on a mathematically-balanced subset of pixels from the training dataset.

## Extracted Features (13 Dimensions)
- **Colour Components**: RGB (3), HSV (3)
- **Vegetation Indices**: ExG, ExR, ExG - ExR (3)
- **Local Statistics**: Local Mean and Local Standard Deviation of the Green channel (2)
- **Smoothed Component**: Gaussian Blurred Green channel (1)
- **Texture Info**: Custom safe `int16` Local Binary Pattern (LBP) normalized (1)

## Dependencies
- `numpy`
- `opencv-python` (`cv2`)
- `scikit-learn`
- `joblib`

## Usage
This method trains a fully supervised model, therefore it requires the `train`, `validation`, and `test` data splits to be active.

```bash
# Assuming you are in the project root directory
python method/method3.py
```

## Workflow
1. **Training Phase**: Reads image-mask pairs from `data/train/`, extracts features, samples `2000` pixels per image, and aggregates them for model training.
2. **Validation & Tuning**: Uses `data/validation/` to perform a Grid Search on Random Forest hyperparameters (`n_estimators`, `max_depth`). Optimal model parameters are saved dynamically based on the highest **Validation IoU**.
3. **Testing Phase**: Predicts mask outputs on `data/test/`, applies morphological noise removal, and saves evaluations.

## Input / Output Directory Structure
- **Input Data**: `data/train`, `data/validation`, `data/test`.
- **Outputs** (Saved to `output/method3/`): 
  - `rf_model.joblib`: Serialized trained model weights.
  - `masks/`: Generated binary masks following EWS polarity.
  - `visuals/`: Tri-panel visual summaries.
  - `metrics.json`: Evaluation accuracy metrics (`Precision, Recall, F1, IoU`) and inference latencies.
