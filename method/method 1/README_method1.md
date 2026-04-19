# Method 1: Unsupervised Segmentation (ExG + GrabCut)

## Overview
Method 1 implements an un-supervised approach for wheat plant segmentation. It relies on the Excess Green (ExG) vegetation index to generate a coarse foreground estimation, followed by OpenCV's GrabCut algorithm for iterative boundary refinement. Finally, morphological operations and connected component analysis are applied to filter out noise.

## Features
- **No Training Required**: Fully unsupervised.
- **Vegetation Index**: Uses ExG ($2G - R - B$) to identify green pixels robustly against soil backgrounds.
- **Refinement**: GrabCut algorithm initialized dynamically with Otsu's thresholding.
- **Post-processing**: Morphological opening/closing and small region noise removal.

## Dependencies
- `numpy`
- `opencv-python` (`cv2`)

## Usage
Simply run the script from the project root or the `method/` directory. The script dynamically resolves the absolute paths to the dataset:

```bash
# Assuming you are in the project root directory
python method/method1.py
```

## Input / Output Directory Structure
- **Input Data**: Reads images and their corresponding ground truth masks from `data/test/`.
- **Outputs** (Saved to `output/method1/`): 
  - `masks/`: Binary masks (0 for plant, 255 for background following EWS dataset conventions).
  - `visuals/`: Tri-panel visual comparisons (Original Image | Prediction | Ground Truth).
  - `metrics.json`: Aggregated metrics including Precision, Recall, F1 score, IoU, and inference time per image.
