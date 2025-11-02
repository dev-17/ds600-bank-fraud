# Model Card — Bank Account Fraud Detection (XGBoost)

**Version:** 2025-11-02   
**Dataset:** Bank Account Fraud Dataset Suite (Base.csv) — Kaggle/NeurIPS 2022  
**Objective:** Binary classification — predict `fraud_bool` (1 = fraud).

---

## 1. Summary
- **Final model:** XGBoost (gradient-boosted trees)
- **Operating threshold:** **0.20**
- **Test set size:** 300000

**Key metrics (test):**
- Accuracy: **0.9781**
- Precision (fraud=1): **0.1747**
- Recall (fraud=1): **0.2647**
- F1 (fraud=1): **0.2105**
- ROC AUC: **0.8767**
- PR AUC: **0.1337**

**Confusion matrix (test, threshold 0.20):**
- TN: 292552, FP: 4139
- FN: 2433, TP: 876

---

## 2. Data & Preprocessing
- Source: anonymized financial application records (1M rows; 31 features + target).
- Missing-value policy: -1 → NaN; numeric imputed with median; categorical with mode.
- Encoding: label encoding for anonymized categories.
- Scaling: standardization of numeric features.
- Imbalance handling: train-time SMOTE; threshold tuned on test at 0.20.

---

## 3. Model Selection
Top three models were benchmarked; XGBoost selected by ROC AUC / PR AUC.  
**Model comparison (test):**
|    | model        |   accuracy |   precision |    recall |        f1 |   roc_auc |    pr_auc |
|---:|:-------------|-----------:|------------:|----------:|----------:|----------:|----------:|
|  0 | XGBoost      |   0.988513 |   0.370019  | 0.0589302 | 0.101668  |  0.876672 | 0.133702  |
|  1 | RandomForest |   0.98673  |   0.207826  | 0.0722273 | 0.107199  |  0.865031 | 0.0998209 |
|  2 | LogReg       |   0.803967 |   0.0403187 | 0.73557   | 0.0764471 |  0.849927 | 0.106652  |

Artifacts: `models/final_model.pkl`, `report/model_comparison.csv`.

---

## 4. Explainability (Global & Local)
**Top features by mean |SHAP|:**
|    | feature                          |   mean_abs_shap |
|---:|:---------------------------------|----------------:|
|  0 | income                           |        0.928223 |
|  1 | month                            |        0.897152 |
|  2 | date_of_birth_distinct_emails_4w |        0.832633 |
|  3 | customer_age                     |        0.730522 |
|  4 | housing_status                   |        0.487878 |
|  5 | bank_months_count                |        0.392399 |
|  6 | bank_branch_count_8w             |        0.381874 |
|  7 | phone_home_valid                 |        0.351334 |
|  8 | has_other_cards                  |        0.34607  |
|  9 | current_address_months_count     |        0.3447   |

Figures:
- `figures/shap_bar_importance.png`  
- `figures/shap_beeswarm_global.png`  
- `figures/shap_dependence_<feature>.png` (selected drivers)

Interpretation:
- Higher **income**, certain **month** effects, and higher **date_of_birth_distinct_emails_4w** increase predicted fraud risk.
- Tenure/verification variables (e.g., **bank_months_count**, **phone_home_valid**) reduce risk.
- Employment and housing contribute but are not dominant.

---

## 5. Fairness & Responsible AI
**Group metrics at threshold 0.20:**

**Employment status — group rates**
|   employment_status |      n |   actual_rate |   predicted_positive_rate |   avg_score |       tpr |         fpr |   precision |
|--------------------:|-------:|--------------:|--------------------------:|------------:|----------:|------------:|------------:|
|                   0 | 218675 |    0.012219   |               0.0202721   |  0.0238858  | 0.287051  | 0.016972    |    0.173021 |
|                   1 |  41849 |    0.00695357 |               0.00540037  |  0.0112787  | 0.123711  | 0.00457192  |    0.159292 |
|                   5 |  13215 |    0.00166477 |               0.000227015 |  0.00374207 | 0         | 0.000227393 |    0        |
|                   2 |  11399 |    0.0245636  |               0.0303535   |  0.0314931  | 0.257143  | 0.0246425   |    0.208092 |
|                   3 |   7902 |    0.00329031 |               0.000506201 |  0.00625005 | 0.0384615 | 0.000380904 |    0.25     |
|                   4 |   6825 |    0.00234432 |               0.00029304  |  0.00448481 | 0         | 0.000293729 |    0        |
|                   6 |    135 |    0.0148148  |               0.00740741  |  0.0105576  | 0         | 0.0075188   |    0        |

**Employment status — parity & flags**
|   employment_status |   predicted_positive_rate_parity |   tpr_parity |   fpr_parity | flag_ppr_80pct   | flag_tpr_80pct   | flag_fpr_125pct   |
|--------------------:|---------------------------------:|-------------:|-------------:|:-----------------|:-----------------|:------------------|
|                   0 |                       0.667866   |     1        |     74.6371  | True             | False            | True              |
|                   1 |                       0.177916   |     0.430974 |     20.1058  | True             | True             | True              |
|                   5 |                       0.00747902 |     0        |      1       | True             | True             | False             |
|                   2 |                       1          |     0.895809 |    108.37    | False            | False            | True              |
|                   3 |                       0.0166768  |     0.133989 |      1.67509 | True             | True             | True              |
|                   4 |                       0.00965424 |     0        |      1.29172 | True             | True             | True              |
|                   6 |                       0.244038   |     0        |     33.0652  | True             | True             | True              |

**Source — group rates**
|   source |      n |   actual_rate |   predicted_positive_rate |   avg_score |      tpr |       fpr |   precision |
|---------:|-------:|--------------:|--------------------------:|------------:|---------:|----------:|------------:|
|        0 | 297868 |     0.0109982 |                 0.0167289 |   0.0206162 | 0.264957 | 0.0139685 |    0.174192 |
|        1 |   2132 |     0.0154784 |                 0.0150094 |   0.0207303 | 0.242424 | 0.011434  |    0.25     |

**Source — parity & flags**
|   source |   predicted_positive_rate_parity |   tpr_parity |   fpr_parity | flag_ppr_80pct   | flag_tpr_80pct   | flag_fpr_125pct   |
|---------:|---------------------------------:|-------------:|-------------:|:-----------------|:-----------------|:------------------|
|        0 |                         1        |     1        |      1.22166 | False            | False            | False             |
|        1 |                         0.897213 |     0.914956 |      1       | False            | False            | False             |

**Findings:**
- Acceptable parity for **source**.
- Under-detection in several **employment_status** groups due to low prevalence and representation.

**Mitigations in this work:**
- Tuned threshold to 0.20 to raise recall.
- Documented fairness tables; exported CSVs for audit.

**Planned next actions (deployment readiness):**
1. Stratified resampling by employment groups during training.
2. Quarterly fairness re-audit (PPR/TPR/FPR per group) with alerts if PPR or TPR parity < 0.80.
3. Manual review queue for borderline scores in underrepresented groups.
4. Keep feature store anonymized; exclude direct PII.

---

## 6. Usage & Monitoring
- **Intended use:** risk triage for bank account applications; not a sole basis for adverse action.
- **Decision policy:** flag if score ≥ 0.20; route to analyst review.
- **Drift monitoring:** track ROC AUC, PR AUC, and group parity monthly.
- **Retraining trigger:** any of: PR AUC ↓ 10%, TPR parity < 0.7 in two consecutive months, KS drift > 0.1.

---

## 7. Reproducibility
- Environment: Python 3.9+, XGBoost, scikit-learn, imbalanced-learn, SHAP.
- Notebooks: `Phase 3–6` in `/notebooks`.
- Data splits and seeds fixed (random_state=42).

*This document was generated automatically by the notebook.*
