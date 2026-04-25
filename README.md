# Student Academic Performance Prediction
### Using Logistic Regression and Random Forest with SHAP-Based Interpretability

A machine learning research study focused on early identification of academically at-risk students using routinely collected institutional data.

**Authors:** Madhav · Vipul Sharma · Keshav Singla  
**Supervisor:** Gifty Gupta  
**Institution:** Chitkara University Institute of Engineering and Technology, Punjab, India

---

## Abstract

Academic failure rarely appears without warning — declining attendance, missed assignments, and falling grades are detectable weeks before a final mark is issued. This study investigates whether supervised machine learning applied to routine institutional data can reliably and interpretably identify at-risk students early enough to act.

Two classifiers — Logistic Regression (interpretable baseline) and Random Forest (ensemble method) — were evaluated on the UCI Student Performance dataset (395 records, Portuguese secondary school mathematics). Both achieved **92.4% test accuracy**, with Random Forest outperforming on:

- **Fail-class recall:** 0.96 vs 0.93
- **AUC-ROC:** 0.95
- **Cross-validation stability:** Mean 90.6% (SD ± 0.022)

SHAP analysis confirmed that G2 and G1 (semester grades) are the strongest predictors, followed by prior failures and absenteeism — all of which are already collected by institutions. A practical three-stage prediction–prioritisation–response framework is proposed for deployment.

---

## Dataset

| Property | Details |
|---|---|
| Source | UCI Student Performance Dataset (Cortez & Silva, 2008) |
| Records | 395 students (Portuguese secondary school, Mathematics) |
| Attributes | 33 original → 41 after one-hot encoding |
| Target variable | Binary Pass/Fail derived from G3 (final grade ≥ 10 = Pass) |

| Class | Count | Proportion |
|---|---|---|
| Pass (1) | 265 | 67.1% |
| Fail (0) | 130 | 32.9% |

> Naive majority classifier baseline: **67.1% accuracy**

---

## Methodology

### Preprocessing
- One-hot encoding of all categorical variables (`drop_first=True`) → 41 features
- No feature normalisation for Random Forest (scale-invariant)
- Logistic Regression: `max_iter=1000` for convergence
- G3 removed from feature set to prevent data leakage; G1 and G2 retained as legitimate early-semester predictors

### Train-Test Split
- 80/20 stratified split (train: n=316, test: n=79)
- Fixed `random_state=42` for reproducibility

### Models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, default solver |
| Random Forest | `random_state=42`, default hyperparameters |

### Evaluation
- Stratified 5-fold cross-validation (`StratifiedKFold`, `n_splits=5`)
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Primary metric:** Fail-class recall (missing an at-risk student is costlier than a false alarm)

### Interpretability
- Random Forest impurity-based feature importance (Gini)
- SHAP values via `TreeExplainer` for individual-level explanations

---

## Results

### Test Set Performance (n = 79)

| Model | Accuracy | Fail Recall | Fail Precision | AUC |
|---|---|---|---|---|
| Logistic Regression | 92.4% | 0.93 | 0.86 | 0.98 |
| Random Forest | 92.4% | 0.96 | 0.84 | 0.95 |

Random Forest correctly flagged **26 of 27** at-risk students (vs. 25 for LR). At scale (1,000 students, 33% fail rate), this translates to **~15 more at-risk students correctly identified per year**.

### Cross-Validation — Random Forest

| Fold | Accuracy |
|---|---|
| Fold 1 | 88.61% |
| Fold 2 | 92.41% |
| Fold 3 | 87.34% |
| Fold 4 | 92.41% |
| Fold 5 | 92.41% |
| **Mean** | **90.63%** |
| **Std Dev** | **±2.21%** |

### Top Predictive Features (Random Forest — Impurity-Based)

| Rank | Feature | Description | Importance |
|---|---|---|---|
| 1 | G2 | Second semester grade | 0.3534 |
| 2 | G1 | First semester grade | 0.2025 |
| 3 | absences | School absences | 0.0373 |
| 4 | failures | Past course failures | 0.0312 |
| 5 | Walc | Weekend alcohol use | 0.0173 |

> G1 + G2 alone account for **over 55%** of total feature importance. All top predictors are already collected by institutions as part of routine record-keeping.

---

## Early Intervention Framework

A three-stage framework for practical deployment:

```
Stage 1 — PREDICTION
 ↳ Run model after first major assessment
 ↳ Flag students above a calibrated risk threshold

Stage 2 — PRIORITISATION
 ↳ Rank by predicted failure probability
 ↳ Cross-reference with real-time engagement data (attendance, submissions)
 ↳ Highest risk + declining engagement → outreach priority

Stage 3 — RESPONSE
 ↳ SHAP explanation guides personalised intervention
    • Low G1            → Academic skills support
    • High absenteeism  → Underlying reasons discussion
    • Past failures + low studytime → Planning assistance
 ↳ Model informs — does NOT replace — educator judgment
```

---

## Limitations

- Single dataset from Portuguese secondary schools — generalisation unverified
- Small sample size (n = 395); limited cross-institutional validity
- No formal subgroup fairness analysis — required before any deployment
- Predictive performance was evaluated; intervention efficacy was not tested
- Default hyperparameters used; tuning may improve results

---

## Repository Structure

```
├── dataset/
│   └── student-mat.csv        # UCI Student Performance dataset
├── model_training.ipynb       # Main notebook — full pipeline
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install pandas scikit-learn matplotlib shap
```

### Option A — Google Colab (recommended)
1. Upload `model_training.ipynb` to Google Colab
2. Upload `student-mat.csv` to your Google Drive under:
   `MyDrive/Student_Performance_Research/dataset/student-mat.csv`
3. Run all cells in order (`Runtime → Run all`)

### Option B — Local (Jupyter)
1. Clone the repository and place `student-mat.csv` in a `dataset/` folder
2. Update the file path in Cell 4 to your local path:

```python
path = "dataset/student-mat.csv"
```

3. Launch Jupyter and run the notebook:

```bash
jupyter notebook model_training.ipynb
```

---

## Notebook Walkthrough

| Section | Description |
|---|---|
| 1–2 | Introduction and dataset overview |
| 3–4 | Google Drive mount and data loading (`student-mat.csv`, semicolon-delimited) |
| 5 | Dataset exploration — shape, info, descriptive stats |
| 6 | Target variable engineering — G3 binarised to Pass (≥10) / Fail (<10) |
| 7 | Feature/target separation — G3 dropped to prevent data leakage |
| 8 | Preprocessing — one-hot encoding (`drop_first=True`), 33 → 41 features |
| 9 | Train-test split — 80/20 stratified, `random_state=42` |
| 10 | Logistic Regression — training, prediction, classification report |
| 11 | Random Forest — training, prediction, classification report |
| 12 | 5-fold stratified cross-validation on Random Forest |
| 13 | ROC curve and AUC analysis |
| 14 | Feature importance — top 10 impurity-based scores (bar chart) |
| 15 | SHAP interpretability — TreeExplainer + summary plot |
| 16–19 | Model comparison, intervention framework, ethical considerations, conclusion |

---

## Tech Stack

```
Python 3.x
├── scikit-learn  — Logistic Regression, Random Forest, cross-validation
├── shap          — SHAP values via TreeExplainer
├── pandas        — Data manipulation
├── matplotlib    — Visualisation (ROC curves, feature importance plots)
└── numpy         — Numerical operations
```

---

## References

1. P. Cortez and A. Silva, "Data Mining to Predict Secondary School Student Performance," FUBUTEC 2008, Porto, Portugal, pp. 5–12.
2. C. Romero and S. Ventura, "Educational Data Mining: A Review of the State of the Art," IEEE Trans. SMC, vol. 40, no. 6, pp. 601–618, 2010.
3. S. B. Kotsiantis et al., "Predicting Performance of Students in Distance Learning," Applied AI, vol. 18, no. 5, pp. 411–426, 2004.
4. C. M. Márquez-Vera et al., "Early Dropout Prediction with Data Mining," Expert Systems, vol. 33, no. 1, pp. 107–124, 2016.
5. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, vol. 30, pp. 4765–4774, 2017.

> Full reference list available in the paper.

---

## Citation

```bibtex
@article{student_perf_2024,
  title   = {Student Academic Performance Prediction Using Logistic Regression
             and Random Forest with SHAP-Based Interpretability},
  author  = {Madhav and Vipul Sharma and Keshav Singla},
  advisor = {Gifty Gupta},
  school  = {Chitkara University Institute of Engineering and Technology},
  year    = {2024}
}
```

This project was developed as part of a Bachelor of Engineering (Computer Science) research study at Chitkara University, Punjab, India.
