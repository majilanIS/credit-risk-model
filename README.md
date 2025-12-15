# Bati_Bank_week-4

# Credit Scoring Business Understanding

## Overview

This section summarizes how credit risk concepts and the Basel II Accord affect our credit-scoring project, explains why we must construct a proxy default label from transaction data, and compares the trade-offs between simple interpretable models and complex high-performance models in a regulated financial setting.

---

### 1) How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

Basel II emphasizes accurate measurement, validation, and governance of credit risk. For this project that means our model must be transparent, auditable, and reproducible: every data source, feature transformation, label definition, validation result, and mapping from predicted probability to credit score must be documented. Clear documentation and interpretability reduce regulatory and operational risk, support model validation and capital calculations, and enable governance by risk committees.

---

### 2) Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

**Why a proxy is necessary**: Supervised learning requires a target. Because the eCommerce dataset does not include an official loan default flag, we must construct a proxy (for example: chargebacks/refunds within a defined window, prolonged non-payment after financed orders, or a delinquency window derived from transaction events) so that we can train and validate models.

**Business risks of using a proxy**:

- _Label error & bias_: The proxy may mislabel customers and reflect merchant or regional peculiarities rather than borrower creditworthiness.
- _Concept drift_: Changes in refund, fraud, or merchant processes can make the proxy divergence from true default behavior over time.
- _Regulatory & reputational risk_: Decisions based on an imperfect proxy can lead to unfair declines or approvals and draw regulatory scrutiny.

**Mitigations**: document the proxy definition precisely, run sensitivity analyses with alternative proxies, adopt conservative decision thresholds initially, and monitor performance post-deployment for re-labeling and retraining.

---

### 3) Key trade-offs between simple interpretable models (Logistic Regression with WoE) versus complex high-performance models (Gradient Boosting) in a regulated financial context

**Simple, interpretable models (Logistic + WoE)**:

- Strengths: transparency, ease of explanation to auditors and stakeholders, simpler validation and stability, and easier enforcement of monotonic relationships.
- Weaknesses: may underfit complex nonlinear relationships in behavioral data and require manual feature engineering.

**Complex, high-performance models (Gradient Boosting)**:

- Strengths: typically higher predictive performance, can capture nonlinear interactions and subtle patterns in RFMS-like features.
- Weaknesses: harder to explain and validate, require additional explainability artifacts (SHAP, partial dependence), careful probability calibration, and stricter post-deployment monitoring.

**Recommended pragmatic approach**: develop an interpretable baseline (logistic + WoE) for governance and regulatory review, and a higher-performance GBM as a candidate production model only after adding explainability, calibration, robustness testing, and monitoring safeguards. Maintain the interpretable model in parallel (shadow mode) during evaluation.
---

### Short action checklist (to include in model documentation)

- Record and version the proxy default definition (exact SQL/pseudocode, windows, and exceptions).
- Produce feature provenance (how R, F, M, S were computed and aggregated).
- Log model runs, calibration, and cohort-level performance (AUC, calibration, precision@k) to MLflow.
- Create a model card summarizing intended use, limitations, fairness checks, and monitoring plan.
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
=======

## Task 6 – Model Deployment & Continuous Integration

**Files:**

- `src/api/main.py`
- `src/api/pydantic_models.py`
- `Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`

**Objective:**  
Package the trained model into a containerized FastAPI service and set up a CI/CD pipeline to automate testing and enforce code quality.

---

### Steps

1. **API Development**

   - Built a REST API using **FastAPI** in `src/api/main.py`.
   - The API loads the best model from the **MLflow registry** (Production stage).
   - Implemented a `/predict` endpoint that accepts new customer transaction data (validated with **Pydantic models**) and returns the risk probability.
   - Added a `/health` endpoint for service monitoring.

2. **Containerization**

   - Created a `Dockerfile` to build a lightweight Python 3.12 image with FastAPI and Uvicorn.
   - Configured `docker-compose.yml` to simplify service startup and environment variable management (e.g., `MODEL_NAME`, `MLFLOW_TRACKING_URI`).

3. **CI/CD Pipeline**
   - Added a GitHub Actions workflow in `.github/workflows/ci.yml`.
   - Pipeline runs automatically on every push to the `main` branch.
   - Steps include:
     - **Linting** with `flake8` to enforce code style.
     - **Testing** with `pytest` to validate functionality.
   - Build fails if either linting or tests fail, ensuring code quality before merging.

---

### Run Commands

```bash
# Build and run locally with Docker Compose
docker compose up --build

# Access API docs
http://localhost:8000/docs
```
