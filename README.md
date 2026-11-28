# Telco Customer Churn Prediction (CM2604 Machine Learning, Coursework Part 01)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanithi20250795-iit/telco-churn-cw/blob/main/CM2604_Churn_Prediction.ipynb)

Binary classification: predict whether a telecom customer will **churn** (cancel their service) using **K-Nearest Neighbours (KNN)**, **Logistic Regression** and **Random Forest**.

## 1. Dataset

- **Name:** IBM Telco Customer Churn
- **Source:** [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns (demographics, services, contract and billing information, and the target `Churn`)
- **Class balance:** 73.5% stayed, 26.5% churned (imbalanced)
- A copy is in [`data/Telco-Customer-Churn.csv`](data/Telco-Customer-Churn.csv) (the Kaggle file `WA_Fn-UseC_-Telco-Customer-Churn.csv`, renamed to match the notebook).

## 2. Repository structure

```
telco-churn-cw/
├── CM2604_Churn_Prediction.ipynb   # full workflow: EDA, preprocessing, models, evaluation, fairness audit
├── churn_analysis.py               # the same pipeline as a plain Python script (used in report Appendix A)
├── data/
│   └── Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

## 3. How to run

**Option A: Google Colab (recommended)**
1. Click the **Open in Colab** badge above.
2. Download `data/Telco-Customer-Churn.csv` from this repository.
3. In Colab, click the **folder icon** on the left, then the **upload icon**, and upload `Telco-Customer-Churn.csv` (keep the exact name).
4. In the data-loading cell, use `df = pd.read_csv("Telco-Customer-Churn.csv")`.
5. Choose **Runtime -> Run all**. The grid search takes a few minutes. Results are reproducible (`random_state=42`).

Uploaded files are deleted when the Colab runtime disconnects, so repeat step 3 in a new session.

**Option B: locally**
```bash
git clone https://github.com/sanithi20250795-iit/telco-churn-cw.git
cd telco-churn-cw
pip install -r requirements.txt jupyter
jupyter notebook CM2604_Churn_Prediction.ipynb
```
Locally, set the loading line to `df = pd.read_csv("data/Telco-Customer-Churn.csv")`. To run the script version instead: `python churn_analysis.py`.

## 4. Methodology

| Step | What was done | Why |
|---|---|---|
| EDA | Class balance, churn rate by contract, internet service, payment method, senior citizen, tech support and paperless billing; numeric distributions by churn; correlation heatmap | Understand the data and find churn drivers |
| Cleaning | `TotalCharges` converted to numeric; 11 blanks (all tenure 0, not yet billed) imputed with the training median inside the pipeline; `customerID` dropped; `Churn` encoded 0/1; `SeniorCitizen` recoded Yes/No | Fix data types; remove a non-informative identifier |
| Split | Stratified 80/20 train/test (5,634 / 1,409), `random_state=42`, done before any fitting | Preserve the 26.5% churn ratio; prevent leakage |
| Preprocessing | Median imputation + `StandardScaler` on numeric columns (KNN, Logistic Regression only); `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` on categorical columns, via `ColumnTransformer` + `Pipeline` (39 encoded features) | Scale-sensitive models need scaled inputs; the pipeline is re-fitted inside each CV fold |
| Ethics | `gender` excluded from model inputs (no churn difference, no business justification) but kept for the fairness audit | Data minimisation and fairness |
| Imbalance | `class_weight="balanced"` for Logistic Regression and Random Forest; stratified splits and CV (KNN has no class weighting) | Improve recall on the churn class |
| Tuning | `GridSearchCV`, 5-fold stratified CV, scored by ROC-AUC, on the training set only | Threshold-independent selection without touching the test set |
| Experimental settings | Setting A: default hyperparameters. Setting B: tuned hyperparameters | Compare both settings as required by the brief |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, 5-fold CV AUC, confusion matrices, ROC curves | Accuracy alone is misleading with imbalanced classes |
| Explainability | Random Forest feature importances, Logistic Regression coefficients | Explain the drivers of churn |
| Fairness audit | Recall, false-positive rate and predicted churn rate by `gender` and `SeniorCitizen` for all tuned models | Support the AI ethics discussion |

### Hyperparameter grids

| Model | Grid | Best |
|---|---|---|
| KNN | `n_neighbors` [3, 5, 7, 11, 15, 21, 31], `weights` [uniform, distance], `p` [1, 2] | k=31, p=2 (Euclidean), uniform |
| Logistic Regression | `C` [0.001, 0.01, 0.1, 1, 10, 100], `penalty` [l1, l2], solver liblinear | C=10, L2 |
| Random Forest | `n_estimators` [200], `max_depth` [6, 10, None], `min_samples_leaf` [1, 5, 10], `max_features` [sqrt] | depth=10, min_samples_leaf=10 |

## 5. Results (test set, 1,409 customers)

| Model | Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| KNN | Baseline | 0.758 | 0.546 | 0.521 | 0.534 | 0.778 | 0.779 |
| KNN | Tuned | 0.781 | 0.589 | 0.575 | 0.582 | 0.826 | 0.834 |
| Logistic Regression | Baseline | 0.737 | 0.503 | 0.783 | 0.613 | 0.842 | 0.846 |
| Logistic Regression | Tuned | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 | 0.846 |
| Random Forest | Baseline | 0.763 | 0.547 | 0.628 | 0.585 | 0.820 | 0.821 |
| **Random Forest** | **Tuned** | 0.757 | 0.528 | 0.797 | **0.635** | **0.845** | 0.848 |

## 6. Key findings

- **Best model:** tuned Random Forest (highest F1, recall and ROC-AUC). Logistic Regression is within about 0.004 AUC, which is within CV noise, and is easier to interpret.
- Churn is much higher for **month-to-month contracts** (42.7% vs 2.8% for two-year), **short tenure** (median 10 months for churners vs 38), **fibre-optic** internet (41.9%), **electronic check** payment (45.3%) and customers **without tech support** (41.6%).
- Tuning helped KNN and Random Forest most (Random Forest recall 0.628 to 0.797); Logistic Regression barely changed.
- KNN has the highest accuracy but the weakest recall, a reminder that accuracy is misleading on imbalanced data.

## 7. AI ethics and post-deployment (summary)

- **Fairness:** gender gaps are small. Senior citizens are flagged far more often (Random Forest: 64.4% vs 35.5%) and have about twice the false-positive rate (0.444 vs 0.232), so the model should only support low-stakes actions such as retention offers.
- **Privacy:** identifier removed, data minimisation, access control; Sri Lanka PDPA No. 9 of 2022 and GDPR-style principles for a real deployment.
- **Transparency:** feature importances and coefficients; human in the loop; model card.
- **Monitoring:** performance and drift monitoring (e.g. PSI), recurring fairness audits, scheduled retraining, A/B tests of retention campaigns with a hold-out group, rollback plan.

## 8. Limitations and future work

- **Limitations:** single static snapshot; one train/test split; modest grids; default 0.5 threshold; precision about 0.5; KNN has no imbalance handling; median imputation for the 11 zero-tenure `TotalCharges` values.
- **Future work:** gradient boosting, SMOTE inside the pipeline, cost-based threshold tuning, calibration, SHAP, survival analysis, nested CV with significance tests, validation on newer data.

## 9. Use of AI assistance

In line with the module's **"5: Full AI"** category, an AI assistant (Claude, by Anthropic) was used to help generate and run the analysis code and to help draft the notebook, README and report text. I reviewed and executed the code and can explain every part of it.

## 10. Author

Sanithi Dissanayake, 20250795, CM2604 Machine Learning (IIT), Semester 1
