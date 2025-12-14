# Bati_Bank_Week-4

## Credit Scoring Business Understanding

### Overview

This section summarizes how credit risk concepts and the Basel II Accord affect our credit-scoring project, explains why we must construct a proxy default label from transaction data, and compares trade-offs between simple interpretable models and complex high-performance models in a regulated financial setting.

---

### 1) Basel II Accord & Model Interpretability

Basel II emphasizes accurate measurement, validation, and governance of credit risk. For this project:

- Models must be **transparent, auditable, and reproducible**.
- Every **data source, feature transformation, label definition, validation result, and mapping from predicted probability to credit score** must be documented.
- Clear documentation reduces regulatory and operational risk and supports governance by risk committees.

---

### 2) Why a Proxy Target Variable is Needed

**Necessity of a proxy:** Supervised learning requires a target. The dataset does not include an official default flag, so we create a proxy:

- Examples: chargebacks/refunds, prolonged non-payment, or delinquency windows derived from transactions.

**Business risks of using a proxy:**

- **Label error & bias:** mislabels customers due to merchant or regional peculiarities.
- **Concept drift:** process changes may diverge the proxy from true default.
- **Regulatory & reputational risk:** unfair approvals/declines may attract scrutiny.

**Mitigation:** Document proxy definitions, run sensitivity analyses, use conservative thresholds, and monitor performance over time.

---

### 3) Model Trade-offs

**Simple interpretable models (Logistic Regression + WoE):**

- ✅ Transparent and easy to explain to auditors
- ✅ Simpler validation and stability
- ❌ May underfit complex nonlinear relationships

**Complex models (Gradient Boosting):**

- ✅ Higher predictive performance
- ✅ Captures nonlinear patterns in RFM features
- ❌ Harder to explain, validate, and monitor
- ❌ Needs additional explainability (SHAP, PDP) and calibration

**Pragmatic approach:**

- Maintain a baseline interpretable model for governance.
- Deploy GBM with explainability, calibration, and monitoring safeguards.
- Keep interpretable model in shadow mode for validation.

---

### Short Action Checklist

- Version the **proxy default definition** (SQL/pseudocode, windows, exceptions).
- Track **feature provenance** (R, F, M, S computations).
- Log **model runs, calibration, cohort-level performance** to MLflow.
- Maintain a **model card** summarizing use, limitations, fairness checks, and monitoring.
- Keep MLflow artifacts in `mlruns/` and metadata in `mlflow.db`.

---

## task-3 - Feature Engineering

** File: ** `src/data_processing.py`
**Objective:** Create `processed_transactions.csv` with the proxy target column.

## Task 4 – Proxy Target Variable Engineering

**File:** `src/predict.py`  
**Objective:** Create `final_dataset.csv` with the proxy target column.

**Steps:**

1. Compute RFM metrics per customer.
2. Define thresholds for high-risk/disengaged customers.
3. Create binary target (`1 = high-risk / likely default`, `0 = low-risk`).
4. Save the processed dataset: `final_dataset.csv`.

---

## Task 5 – Model Training & Tracking

**File:** `src/train.py`  
**Objective:** Train models and track experiments using MLflow.

**Run Command:**

```bash
python src/train.py
```
