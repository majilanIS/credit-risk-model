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

_End of section._
