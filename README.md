# Urban Waste Detection with YOLOv8

**Automated detection and statistical analysis of urban waste using deep learning**

Bachelor's Thesis in Mathematics · Universidad de La Laguna · May 2026  
*Distinction (Matrícula de Honor)*

---

## Overview

Urban waste pollution is a growing environmental concern that demands objective, scalable measurement tools. Traditional monitoring relies on manual inspection — neither efficient nor reproducible at city scale.

This project builds an automatic waste detection system using **YOLOv8** trained via transfer learning on 18 waste categories. Beyond standard object detection, the model's outputs are analysed through **statistical inference**, introducing **surface density** as a more robust quantitative indicator of urban contamination than simple object counting.

The work bridges two roles: data engineering (dataset structuring, compute environment, pipeline design) and data science (model selection, training, metric evaluation, inferential statistical analysis).

---

## Results

The optimised model (1,000 epochs, full validation set) achieved the following metrics on an independent test set of 100 images:

| Metric | Value |
|--------|-------|
| Precision | **0.844** |
| Recall | **0.774** |
| mAP50 | **0.814** |
| mAP50-95 | **0.646** |

**Key statistical finding:** despite detecting 104 fewer objects than ground truth (290 vs. 394), the model estimated mean surface density with an error of only **0.21 %** (12.10 % predicted vs. 11.89 % real). A 95 % confidence interval via log-transformation and Student's t-distribution places the true mean surface density between **2.67 % and 5.12 %**.

This demonstrates that surface density is a statistically more robust contamination metric than object count — small objects (cigarettes, broken glass) inflate counts but contribute negligible area.

---

## Dataset

- **Source:** [Trash Detection Image Dataset](https://www.kaggle.com/datasets/ahnaftahmeed/trash-detection-image-dataset) (Kaggle)
- **Total images:** ~6,000 labelled images
- **Split:** 4,200 train / 1,700 validation / 100 test
- **Classes:** 18 urban waste categories

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Aluminium foil | 9 | Other waste |
| 1 | Bottle | 10 | Other plastics |
| 2 | Bottle cap | 11 | Paper |
| 3 | Broken glass | 12 | Plastic bag / wrapper |
| 4 | Can | 13 | Plastic container |
| 5 | Cardboard | 14 | Ring pull |
| 6 | Cigarette | 15 | Straw |
| 7 | Cup | 16 | White cork |
| 8 | Lid | 17 | Unlabelled waste |

Class distribution is imbalanced (reflecting real-world frequencies). No rebalancing techniques were applied deliberately — artificially modifying the distribution would bias the model away from realistic contamination patterns.

---

## Methodology

### Model
- **Architecture:** YOLOv8 (Ultralytics), single-pass real-time object detection
- **Strategy:** Transfer learning from COCO pretrained weights + fine-tuning on waste dataset

### Experiments
Three successive experiments were run to study the effect of training length and validation set integrity:

| Experiment | Epochs | Validation set | mAP50 (val) | Notes |
|------------|--------|----------------|-------------|-------|
| 1 | 80 | 1,700 images | 0.386 | Underfitting baseline |
| 2 | 1,000 | 250 images | 0.027 | Methodological error — incomplete validation |
| 3 | 1,000 | 1,700 images | **0.437** | Optimised model |

Experiment 2 illustrates in practice how an unrepresentative validation sample completely distorts metrics — a relevant lesson in experimental rigour.

### Statistical Analysis
Surface density per image is defined as:

```
Ds = Σ (wi × hi)   [for ground truth]
Ds = Σ |x2-x1| × |y2-y1| / (W × H)   [for predictions]
```

The density variable presents positive skewness. A log-transformation was applied and normality was verified via Shapiro-Wilk test (p = 0.169) and QQ-plot. A 95 % confidence interval was computed using Student's t-distribution with n−1 = 99 degrees of freedom.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Deep learning | PyTorch, Ultralytics YOLOv8 |
| Data processing | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Dataset management | Roboflow |
| Compute | Google Colab (GPU), Jupyter Notebook |

---

## Repository Structure

```
tfg-urban-waste-detection/
├── notebooks/
│   └── Object detection using YOLOv8-resumido.ipynb   # Full pipeline
├── results/
│   ├── results_tables.xlsx                             # Metrics across experiments
│   ├── comparativa_inferencia.png                      # Density inference plot
│   └── diagnostico_normalidad.png                      # Normality diagnostics
├── data.yaml                                           # Dataset configuration
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install ultralytics torch numpy pandas matplotlib seaborn
```

### Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pretrained weights
model.train(data="data.yaml", epochs=1000, imgsz=416)
```

### Inference

```python
model = YOLO("best.pt")  # Load fine-tuned weights
results = model("your_image.jpg")
results[0].show()
```

---

## Key Findings

- **Plastics dominate by area:** plastics represent 41.4 % of detected objects but occupy **56.2 % of total contaminated surface** — making them the primary target for urban cleaning interventions.
- **Count is a misleading metric:** cigarettes (14 % of objects) and broken glass (12.9 %) contribute negligible surface area (1.3 % and 0.2 % respectively), despite their high frequency.
- **Surface density is scale-invariant:** the model underestimated object count by 26 % but estimated surface density with only 0.21 % error, confirming density as a more actionable indicator for waste management policy.

---

## Future Work

- Interactive Power BI dashboard integrating real-time detection data with geographic waste distribution
- Clustering analysis (k-means) to identify co-occurrence patterns between waste categories
- Deployment as a lightweight REST API (FastAPI) for integration with city monitoring systems

---

## Citation

If you use this work, please cite:

```
León Marquillas, Ágatha. (2026). Mathematical foundations of Deep Learning and its applications.
Bachelor's Thesis, Universidad de La Laguna.
```

---

## License

This project is for academic and research purposes.  
Dataset credit: [ahnaftahmeed on Kaggle](https://www.kaggle.com/datasets/ahnaftahmeed/trash-detection-image-dataset)
