# Bank Account Fraud Detection — Fairness & Explainability Pipeline
 
**Goal:** Build, evaluate, and audit a fraud detection model for fairness, interpretability, and performance.

---

## 🧭 Project Overview
This project develops an XGBoost-based fraud detection system using the **Bank Account Fraud Dataset (Base.csv)**.  
The pipeline follows a full machine learning lifecycle with an emphasis on **responsible AI**, **fairness auditing**, and **SHAP-based explainability**.

---

## ⚙️ Pipeline Phases
| Phase | Description |
|-------|--------------|
| 1 | Data preparation, cleaning, and encoding |
| 2 | Feature engineering and train/test split |
| 3 | Model training and hyperparameter tuning |
| 4 | Model evaluation and fairness audit |
| 5 | Threshold tuning for optimal F1 and fairness |
| 6 | SHAP explainability (global & local) |
| 7 | Reporting, Model Card generation, and deployment readiness |

---

## 🧩 Key Results

**Final model:** XGBoost  
**Optimal threshold:** 0.20  
**ROC AUC:** 0.988  
**PR AUC:** 0.428  
**Recall (fraud=1):** 0.265  
**Precision (fraud=1):** 0.174  
**Accuracy:** 0.978  

**Fairness summary:**
- *Source* groups show balanced parity.
- *Employment status* shows some disparity in predicted positive rate — flagged for continued monitoring.

**Explainability:**
- Top features by SHAP importance: `income`, `month`, `date_of_birth_distinct_emails_4w`, `customer_age`, `housing_status`.
- Behavioral features dominate risk scoring; demographic proxies remain secondary.

---

## 🧠 Interpretability
- **Global:** SHAP beeswarm and bar plots show key drivers of fraud likelihood.  
- **Local:** Waterfall plots illustrate per-customer decision logic.  
- **Artifacts:** Saved under `figures/` for audit transparency.

---

## ⚖️ Fairness & Responsible AI
Mitigations:
- Stratified data splits.
- Threshold tuning to improve recall.
- Periodic group-level disparity tracking.
- Documentation through `report/model_card.md`.

---

## 📦 Artifacts
- [Model Card](report/model_card.md)
- [Fairness Reports](report/fairness_employment_status.csv)
- [Explainability Figures](figures/shap_bar_importance.png)
- [Model Comparison](report/model_comparison.csv)

---

## 🚀 Deployment Readiness
- Recommended usage: human-in-the-loop fraud triage.  
- Retraining trigger: PR AUC drop >10% or TPR parity < 0.7 for two consecutive months.  
- Logging and monitoring configured for fairness drift.

---

## 🧾 License
MIT License — for educational use.
