### 4. Experimental Results (E)

#### 4.1 Dataset Split and Setup

The experiments are conducted on the EWS (Eschikon Wheat Segmentation) dataset, which consists of RGB images of wheat fields with corresponding pixel-level ground truth masks.

To ensure a fair evaluation, the dataset is split into training, validation, and test sets. For supervised and machine learning methods (Method 2–4), a standard split is adopted (e.g., 70% training, 15% validation, 15% test). Method 1 (Unsupervised) does not require training and is evaluated directly on the test set.

* **Method 1 (Unsupervised)**: No training phase; evaluation is performed directly on the test set.
* **Method 2 (SVM)** and **Method 3 (Random Forest)**:

  * Training set is used to learn pixel-level classifiers.
  * Validation set is used for parameter tuning.
  * Test set is used for final evaluation.
* **Method 4 (U-Net)**:

  * A supervised deep learning model trained on the training set.
  * Due to computational constraints, only a single training run is conducted.

All methods are evaluated on the same test set to ensure comparability.

Images are consistently preprocessed across methods, including resizing and normalization. Classical methods rely on handcrafted features, while deep learning directly uses raw pixel inputs.

---

#### 4.2 Evaluation Metrics

To evaluate segmentation performance, we use standard pixel-level metrics:

* **Precision**: Proportion of correctly predicted plant pixels among all predicted plant pixels.
* **Recall**: Proportion of correctly predicted plant pixels among all ground truth plant pixels.
* **F1-score**: Harmonic mean of precision and recall.
* **Intersection over Union (IoU)**: Measures overlap between predicted masks and ground truth masks.

Additionally, we report:

* **Inference time (seconds)**: Average time required to process one image.

For Methods 1–3, both mean and standard deviation are reported across the test set. For Method 4, only mean values are reported due to a single training run.

---

#### 4.3 Quantitative Comparison

| Method   | Precision     | Recall        | F1-score      | IoU           | Inference Time (s) |
| -------- | ------------- | ------------- | ------------- | ------------- | ------------------ |
| Method 1 | 0.494 ± 0.243 | 0.718 ± 0.197 | 0.548 ± 0.215 | 0.406 ± 0.192 | 0.456              |
| Method 2 | 0.718 ± 0.177 | 0.857 ± 0.163 | 0.771 ± 0.152 | 0.650 ± 0.181 | 14.790             |
| Method 3 | 0.863 ± 0.186 | 0.710 ± 0.262 | 0.757 ± 0.224 | 0.651 ± 0.233 | 0.093              |
| Method 4 | 0.847         | 0.822         | 0.823         | 0.719         | 0.015              |

From the quantitative results, several key observations can be made:

* **Method 4 (U-Net)** achieves the best overall performance, with the highest F1-score (0.823) and IoU (0.719), demonstrating the effectiveness of deep learning for segmentation tasks.
* **Method 3 (Random Forest)** achieves competitive performance with significantly lower inference time, making it a strong practical alternative.
* **Method 2 (SVM)** achieves the highest recall but suffers from very high inference time, limiting scalability.
* **Method 1 (Unsupervised)** performs the worst overall, especially in precision, indicating limitations of heuristic-based approaches.

Overall, deep learning methods achieve superior accuracy, while classical machine learning methods offer better efficiency and simpler deployment, highlighting a clear trade-off between performance and computational cost.

---

#### 4.4 Qualitative Examples

Qualitative results are shown in Figure X, where segmentation outputs from all four methods are compared on representative test images.

* **Method 1** often fails under complex lighting conditions due to reliance on color-based heuristics, leading to noisy or incomplete segmentation.
* **Method 2** produces smoother segmentation but may lose fine details due to feature abstraction and classifier limitations.
* **Method 3** achieves more stable segmentation, benefiting from ensemble learning and better generalization across structured plant regions.
* **Method 4 (U-Net)** generates the most accurate and visually consistent segmentation, as it learns hierarchical features directly from data.

Failure cases are also observed:

* **Over-segmentation** in dark soil regions, where background pixels are misclassified as plants.
* **Under-segmentation** under low contrast conditions, where plant boundaries are difficult to distinguish.

These observations highlight the limitations of traditional methods and demonstrate the robustness and adaptability of deep learning approaches.
