# Method 4: U-Net Deep Learning Segmentation

## Overview
Method 4 is the deep-learning counterpart of Method 3. It replaces the handcrafted-feature + Random Forest pipeline with an end-to-end U-Net convolutional network that learns the wheat / soil boundary directly from raw RGB pixels. The same train / validation / test splits and the same EWS label polarity are used, so results are directly comparable with Method 3.

## Model
- **Architecture**: Custom 4-level U-Net (encoder-decoder with skip connections), `base=32` channels, BatchNorm + ReLU, transposed convolutions for upsampling.
- **Input**: RGB image, normalised to `[0, 1]`, shape `(3, H, W)`.
- **Output**: Single-channel logit map; `sigmoid > 0.5` produces the binary plant mask.

## Loss and Optimisation
- **Loss**: 0.5 * Binary Cross Entropy + 0.5 * Dice loss (handles class imbalance and sharpens boundaries).
- **Optimiser**: Adam, `lr=1e-3`, `weight_decay=1e-5`.
- **Scheduler**: Cosine annealing over all epochs.
- **Model selection**: Best weights by validation IoU are kept.

## Data Augmentation (train split only)
- Random horizontal and vertical flips
- Random 90-degree rotations
- Mild brightness jitter (0.85x - 1.15x)

Augmentation is disabled on the validation and test splits.

## Dependencies
- `numpy`
- `opencv-python` (`cv2`)
- `torch` (PyTorch, with CUDA / MPS / CPU auto-detected)

## Usage
```bash
# From the project root directory
python "method/method 4/method4.py"
```

Default hyperparameters: `epochs=60`, `batch_size=8`, `lr=1e-3`. Adjust in the `__main__` block if needed.

## Workflow
1. **Training Phase**: Loads image-mask pairs from `data/train/` with augmentation, iterates for `epochs` epochs, evaluates IoU on `data/validation/` after each epoch, and keeps the best-IoU weights.
2. **Model Saving**: The best model state dict is stored at `output/method4/unet_model.pt`.
3. **Testing Phase**: Runs inference on `data/test/`, applies light morphological open + close for consistency with Method 3, and computes Precision / Recall / F1 / IoU per image.

## Input / Output Directory Structure
- **Input Data**: `data/train`, `data/validation`, `data/test`.
- **Outputs** (saved to `output/method4/`):
  - `unet_model.pt`: PyTorch state dict of the best U-Net model.
  - `masks/`: Predicted binary masks in EWS polarity (black = plant).
  - `visuals/`: Tri-panel visuals `Original | Prediction | Actual`.
  - `metrics.json`: Averaged `Precision, Recall, F1, IoU` plus training time and mean inference latency.

## Why Deep Learning Here
Method 3 relies entirely on local handcrafted descriptors (RGB / HSV / ExG / LBP / local statistics) and can fragment thin leaves or miss faint early-stage seedlings. U-Net learns multi-scale context through its contracting / expanding path and skip connections, so it can model both the fine leaf boundaries and the global plant shape, which directly addresses the failure modes observed in Method 3.
