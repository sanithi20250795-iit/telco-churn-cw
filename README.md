# Telco Customer Churn Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanithi20250795-iit/telco-churn-cw/blob/main/churn_prediction.ipynb)

Binary classification coursework: predict whether a telecom customer will **churn** (cancel their service) using **K-Nearest Neighbours (KNN)**, **Logistic Regression** and **Random Forest**.

## 1. Dataset

- **Name:** IBM Telco Customer Churn
- **Source:** [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns (demographics, services, contract and billing information, and the target `Churn`)
- **Class balance:** 73.5% stayed, 26.5% churned (imbalanced)
- A copy of the file is in [`data/`](data/WA_Fn-UseC_-Telco-Customer-Churn.csv).

## 2. Repository structure

```
telco-churn-cw/
├── churn_prediction.ipynb                 # full workflow: EDA, preprocessing, models, evaluation, ethics
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model_results.csv                      # metrics table produced by the notebook
├── requirements.txt
└── README.md
```

## 3. How to run

**Option A: Google Colab (recommended)**
1. Click the **Open in Colab** badge above.
2. Choose **Runtime -> Run all**.
3. When the upload button appears in the first code cell, select `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (download it from this repository first).
4. The grid search takes a few minutes. Results are reproducible (`random_state=42`).

**Option B: locally**
```bash
git clone https://github.com/sanithi20250795-iit/telco-churn-cw.git
cd telco-churn-cw
pip install -r requirements.txt jupyter
jupyter notebook churn_prediction.ipynb
```
For a local run, replace the `files.upload()` lines in the first code cell with `df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")` (the `google.colab` module only exists in Colab).

## 4. Methodology

| Step | What was done | Why |
|---|---|---|
| EDA | Distributions, churn rate by contract, tenure, charges, internet service, payment method, tech support, correlation heatmap | Understand the data and find churn drivers |
| Cleaning | `TotalCharges` converted to numeric; 11 blanks (all tenure 0, not yet billed) set to 0; `customerID` dropped; `Churn` encoded 0/1 | Fix data types and remove non-informative identifier |
| Split | Stratified 80/20 train/test, `random_state=42` | Preserve the 26.5% churn ratio; reproducibility |
| Preprocessing | `StandardScaler` on numeric columns, `OneHotEncoder(drop='first', handle_unknown='ignore')` on categorical columns, inside a `ColumnTransformer` + `Pipeline` | KNN and Logistic Regression need scaled inputs; the pipeline prevents data leakage |
| Imbalance | `class_weight='balanced'` (Logistic Regression, Random Forest); stratified splits and CV | Improve recall on the minority (churn) class |
| Tuning | `GridSearchCV`, 5-fold stratified CV, scored by ROC-AUC, on the training set only | Fair, threshold-independent model selection without touching the test set |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC and precision-recall curves | Accuracy alone is misleading with imbalanced classes |
| Explainability | Random Forest feature importance, Logistic Regression coefficients | Understand and explain the drivers of churn |
| Fairness audit | Metrics split by `gender` and `SeniorCitizen` | Support the AI ethics discussion |

### Hyperparameter grids

| Model | Grid |
|---|---|
| KNN | `n_neighbors` [5, 11, 21, 31], `weights` [uniform, distance], `metric` [euclidean, manhattan] |
| Logistic Regression | `C` [0.01, 0.1, 1, 10], `penalty` [l1, l2], `solver` liblinear |
| Random Forest | `n_estimators` [200, 400], `max_depth` [5, 10, None], `min_samples_split` [2, 10], `min_samples_leaf` [1, 4] |

Best parameters found: KNN (`k=31`, euclidean, uniform), Logistic Regression (`C=10`, L1), Random Forest (`max_depth=10`, `min_samples_leaf=4`, `min_samples_split=10`, `n_estimators=400`).

## 5. Results (test set, 1,409 customers)

| Model | Setting | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| KNN | Baseline | 0.763 | 0.554 | 0.551 | 0.552 | 0.793 |
| KNN | Tuned | 0.795 | 0.620 | 0.586 | 0.602 | 0.833 |
| Logistic Regression | Baseline | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 |
| Logistic Regression | Tuned | 0.737 | 0.503 | 0.783 | 0.613 | 0.841 |
| Random Forest | Baseline | 0.791 | 0.639 | 0.492 | 0.556 | 0.824 |
| **Random Forest** | **Tuned** | 0.766 | 0.541 | 0.770 | **0.636** | **0.844** |

## 6. Key findings

- **Best model:** tuned Random Forest (highest F1 and ROC-AUC). Logistic Regression is almost as good and easier to interpret.
- Churn is much higher for **month-to-month contracts** (42.7% vs 2.8% for two-year), **short tenure** (median 10 months for churners vs 38), **fiber optic** internet (41.9%), **electronic check** payment (45.3%) and customers **without tech support** (41.6%).
- Tuning improved KNN's ROC-AUC (0.793 to 0.833) and greatly raised Random Forest's recall (0.492 to 0.770). Accuracy fell slightly for Random Forest because the tuned model flags more customers, which is acceptable when catching churners matters more.

## 7. AI ethics and post-deployment (summary)

- **Fairness:** gender shows no meaningful difference; senior citizens churn more and are flagged far more often (64% vs 33%), so group metrics must be monitored and offers designed to help, not penalise.
- **Privacy:** data minimisation, anonymised identifiers, access control, GDPR-style principles.
- **Transparency:** feature importance and coefficients (SHAP as future work); human in the loop.
- **Monitoring:** data drift, concept drift, per-group fairness, scheduled retraining, A/B tests of retention campaigns, rollback plan.

Full discussion is in section 8 of the notebook and in the report.

## 8. Limitations and future work

- **Limitations:** single static snapshot; class imbalance; one test split (noisy comparisons); some best hyperparameters lie at the edge of the grids; default 0.5 threshold; precision about 0.5.
- **Future work:** gradient boosting (XGBoost/LightGBM), SMOTE inside the pipeline, cost-based threshold tuning, SHAP, survival analysis, wider grids, nested CV, calibration.

## 9. Use of AI assistance

In line with the module's **"5: Full AI"** category, I used an AI assistant (Claude, by Anthropic) while working on this coursework.

**How AI was used:**
- Explaining the coursework brief and planning the workflow.
- Suggesting the structure of the notebook and drafting code for the preprocessing pipeline, model training, hyperparameter tuning and evaluation.
- Drafting explanatory text for the notebook and README, and helping me understand the concepts (data leakage, cross-validation, evaluation metrics).

**What I did myself:**
- Ran all code in Google Colab and checked the outputs.
- Ran the exploratory analysis and checked that every figure quoted in the notebook and report matches my own outputs.
- Corrected and adapted the code and text, and managed the project through Git.
- Wrote the final report and can explain every part of the code and results.

## 10. Author

Sanithi Dissanayke , 200250795,CM2604 - Machine Learning (IIT) - Semester 1
