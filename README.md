# 26T1-COMP9517-wheat-segmentation-cv
A computer vision project for comparing multiple segmentation methods on wheat crop images.

This project aims to develop and compare multiple computer vision methods for segmenting wheat crops from field images.

We implement and evaluate different approaches, including traditional computer vision methods, machine learning models, and deep learning architectures. The goal is to analyze their performance under various conditions and understand their strengths and limitations.

The project is based on the EWS (Eschikon Wheat Segmentation) dataset, and performance is evaluated using metrics such as IoU, precision, recall, and F1-score.

## Setup

We recommend using Python 3.10.

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the **EWS (Eschikon Wheat Segmentation)** dataset, which is designed for crop segmentation tasks in real-world agricultural environments.

- The dataset contains **190 RGB images** (350 × 350 resolution)
- Each image is paired with a **manually annotated binary mask**
  - 1 → crop (wheat)
  - 0 → background (soil)

The dataset is already split into **training, validation, and test sets**, and it is important not to mix these subsets during experiments to avoid data leakage. :contentReference[oaicite:0]{index=0}  

---

### 🔗 Download

The dataset can be downloaded from:

https://www.research-collection.ethz.ch/entities/researchdata/165d22fc-6b0f-4fc3-a441-20d8bdc50a70

---

### ⚠️ Notes

- The dataset is **not included in this repository** due to size constraints  
- Please download and extract it locally under the `data/` directory  
- Make sure to follow the correct train/validation/test split when running experiments  

---

### 📁 Expected Data Structure

After extraction, the dataset should be organized as follows:

data/
train/
images/
masks/
val/
images/
masks/
test/
images/
masks/