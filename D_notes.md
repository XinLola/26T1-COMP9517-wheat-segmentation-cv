# D Workbench

## 1. Clean Test Overall Results

These results are obtained from the unified evaluation script on the test set.

| Method | Precision | Recall | F1-score | IoU | Notes |
|---|---:|---:|---:|---:|---|
| method1 | 0.4766 | 0.7234 | 0.5384 | 0.3951 | Higher recall, but many false positives |
| method2 | 0.7075 | 0.8479 | 0.7583 | 0.6358 | Strong recall and balanced overall performance |
| method3 | 0.8626 | 0.7099 | 0.7574 | 0.6508 | Highest precision and best IoU |

### Quick interpretation
- method1 performs the worst overall.
- method2 and method3 are close in F1-score.
- method3 achieves the highest precision and IoU.
- method2 achieves the highest recall.
- This suggests that method3 is more conservative but cleaner, while method2 captures more wheat pixels.

---

## 2. Best and Worst Cases

### method1
**Worst 3**
- FPWW0180375_RGB1_20170222_133519_6.png | IoU = 0.0215 | F1 = 0.0421
- FPWW0180374_RGB1_20170222_133502_6.png | IoU = 0.0311 | F1 = 0.0604
- FPWW0180094_RGB1_20170308_114711_6.png | IoU = 0.1370 | F1 = 0.2410

**Best 3**
- FPWW0180367_RGB1_20170405_173248_6.png | IoU = 0.6176 | F1 = 0.7636
- FPWW0180151_RGB1_20170420_130936_6.png | IoU = 0.6309 | F1 = 0.7737
- FPWW0180546_RGB1_20170505_135217_6.png | IoU = 0.6700 | F1 = 0.8024

### method2
**Worst 3**
- FPWW0180374_RGB1_20170222_133502_6.png | IoU = 0.1623 | F1 = 0.2793
- FPWW0180741_RGB1_20170308_135245_6.png | IoU = 0.3360 | F1 = 0.5030
- FPWW0180375_RGB1_20170222_133519_6.png | IoU = 0.4020 | F1 = 0.5735

**Best 3**
- FPWW0180151_RGB1_20170420_130936_6.png | IoU = 0.8573 | F1 = 0.9231
- FPWW0180367_RGB1_20170505_163137_6.png | IoU = 0.8705 | F1 = 0.9308
- FPWW0180546_RGB1_20170505_135217_6.png | IoU = 0.9024 | F1 = 0.9487

### method3
**Worst 3**
- FPWW0180374_RGB1_20170222_133502_6.png | IoU = 0.0000 | F1 = 0.0000
- FPWW0180375_RGB1_20170222_133519_6.png | IoU = 0.1647 | F1 = 0.2827
- FPWW0180755_RGB1_20170323_173949_6.png | IoU = 0.3369 | F1 = 0.5040

**Best 3**
- FPWW0180367_RGB1_20170505_163137_6.png | IoU = 0.8839 | F1 = 0.9384
- FPWW0180151_RGB1_20170420_130936_6.png | IoU = 0.9136 | F1 = 0.9548
- FPWW0180546_RGB1_20170505_135217_6.png | IoU = 0.9138 | F1 = 0.9550

### Common observations
- FPWW0180374_RGB1_20170222_133502_6.png appears in the worst cases of all three methods.
- FPWW0180375_RGB1_20170222_133519_6.png also appears repeatedly as a difficult image.
- FPWW0180151_RGB1_20170420_130936_6.png and FPWW0180546_RGB1_20170505_135217_6.png are consistently segmented well.

---

## 3. Failure Pattern Notes

### Common failure types
1. **Complex soil texture**
   - Some background soil regions have similar colour or texture to wheat, causing false positives.

2. **Low contrast / shadow**
   - Under darker illumination or strong shadow, the boundaries between wheat and background become unclear.

3. **Thin structures**
   - Small leaves and thin plant regions are easily missed, leading to lower recall.

4. **Fragmented masks**
   - Some methods produce broken foreground regions or small holes inside the wheat mask.

### Likely explanation for the hardest cases
- FPWW0180374_RGB1_20170222_133502_6.png is likely extremely challenging because the foreground is weak and the background appearance overlaps with wheat appearance.
- FPWW0180375_RGB1_20170222_133519_6.png likely contains unclear wheat boundaries and difficult texture separation.



---

## 4. Timing Summary

| Method | Training Time | Testing Time | Status |
|---|---:|---:|---|
| method1 | N/A or 0 | 1.7423 s/image | confirmed |
| method2 | 16.07 s | 51.04 s/image (reported by B, unit to be re-checked) | from B's experiment_summary.csv |
| method3 | 18.82 s (gaussian noise run) / 25.23 s (clean run) | 0.1240 s/image (clean) / 0.137 s/image (gaussian noise) | confirmed |

### Important note
Do **not** use the `test_time_sec_mean` from `output/D_eval/summary.csv` as the real testing time of the methods. That value only reflects the speed of the evaluation script reading masks and computing metrics, not actual model inference time.
For method2, the reported testing time is taken from B's experiment summary and should be double-checked for unit consistency before final submission.

---

## 5. Robustness Progress

### Completed
- Distorted test images have been generated for:
  - gaussian_noise
  - gaussian_blur
  - low_brightness
  - low_contrast
  - occlusion
- Original test masks have been copied into each distorted directory.
- method1 has been re-run on gaussian_noise and all 24 prediction masks were generated successfully.
- method3 has also been re-run on gaussian_noise and all 24 prediction masks were generated successfully.
- method2 gaussian_noise robustness results were incorporated from B's experiment outputs.

### Current status
- The Gaussian noise robustness pipeline is now complete for all three methods.
- method1 and method3 robustness results were obtained using the unified evaluation workflow.
- method2 robustness information is currently based on B's reported experiment outputs.

### Next actions
1. Finalise the robustness comparison table for Gaussian noise.
2. If time permits, extend the same analysis to one additional distortion such as gaussian_blur or low_brightness.
3. Verify the exact unit and definition of method2 testing time before the final report submission.

## 6. Robustness result: Gaussian noise

### method1 on gaussian_noise
| Method | Distortion | Precision | Recall | F1-score | IoU | Valid |
|---|---|---:|---:|---:|---:|---|
| method1 | gaussian_noise | 0.1506 | 0.0162 | 0.0209 | 0.0134 | yes |

### method2 on gaussian_noise
| Method | Distortion | Precision | Recall | F1-score | IoU | Valid |
|---|---|---:|---:|---:|---:|---|
| method2 | gaussian_noise | 0.9383 | 0.6922 | 0.7833 | 0.6544 | from B |

### method3 on gaussian_noise
| Method | Distortion | Precision | Recall | F1-score | IoU | Valid |
|---|---|---:|---:|---:|---:|---|
| method3 | gaussian_noise | 0.8386 | 0.1958 | 0.2877 | 0.1923 | yes |

### Comparison with clean test
| Method | Setting | Precision | Recall | F1-score | IoU |
|---|---|---:|---:|---:|---:|
| method1 | clean | 0.4766 | 0.7234 | 0.5384 | 0.3951 |
| method1 | gaussian_noise | 0.1506 | 0.0162 | 0.0209 | 0.0134 |
| method2 | clean | 0.7075 | 0.8479 | 0.7583 | 0.6358 |
| method2 | gaussian_noise | 0.9383 | 0.6922 | 0.7833 | 0.6544 |
| method3 | clean | 0.8626 | 0.7099 | 0.7574 | 0.6508 |
| method3 | gaussian_noise | 0.8386 | 0.1958 | 0.2877 | 0.1923 |

### Observations
- method1 is extremely sensitive to Gaussian noise and almost collapses under corruption.
- method3 is more robust than method1, but its recall drops sharply under noise, meaning it becomes conservative and misses many wheat regions.
- According to B's reported robustness results, method2 remains the most stable under Gaussian noise and retains strong overall performance.


### Preliminary robustness conclusion
Under Gaussian noise, method1 is the least robust and almost collapses. Method3 is substantially more robust than method1, but its recall decreases sharply, indicating conservative segmentation under corruption. Based on the currently available results, method2 appears to be the most stable method under Gaussian noise.

## 7. Method2 failure analysis from B

Representative difficult samples include:
- FPWW0180367_RGB1_20170412_145913_6
- FPWW0180049_RGB1_20170420_111709_6
- FPWW0180049_RGB1_20170517_142522_6
- FPWW0180755_RGB1_20170323_173949_6
- FPWW0180049_RGB1_20170329_143149_6

Reported main reason:
- complex background and insufficient feature separability

### Consistency note
For the final cross-method comparison on the clean test set, the metrics reported in this document use the unified evaluation script to ensure fairness across methods. For method2, additional timing, robustness, and failure-analysis information is incorporated from the method owner's experiment outputs.