# ==============================
# Model Training & Tracking
# ==============================

# 1️⃣ SETUP: Import libraries
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==============================
# 2️⃣ DATA PREPARATION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "../data/processed/processed_transactions.csv"
)

df = pd.read_csv(DATA_PATH)
TARGET = "FraudResult"

# 🚨 DROP CustomerId (identifier, not a feature)
X = df.drop(columns=[TARGET, "CustomerId"])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 3️⃣ MODEL SELECTION
# ==============================

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
}

# ==============================
# 4️⃣ EXPERIMENT TRACKING (MLflow)
# ==============================

mlflow.set_experiment("Credit_Risk_Modeling")

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # ==============================
        # 5️⃣ MODEL EVALUATION
        # ==============================

        mlflow.log_params(model.get_params())

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        # Save model
        mlflow.sklearn.log_model(model, "model")

# ==============================
# 6️⃣ HYPERPARAMETER TUNING
# ==============================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring="f1",
    cv=3
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

# ==============================
# 7️⃣ REGISTER BEST MODEL
# ==============================

with mlflow.start_run(run_name="Best_RandomForest"):

    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    mlflow.log_params(best_rf.get_params())
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

    mlflow.sklearn.log_model(
        best_rf,
        artifact_path="model",
        registered_model_name="CreditRiskModel"
    )
