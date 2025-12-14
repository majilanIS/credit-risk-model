# =========================
# data_processing.py
# =========================

import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =========================
# File Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "../data/raw/data.csv")
PROCESSED_PATH = os.path.join(
    BASE_DIR, "../data/processed/processed_transactions.csv"
)

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# =========================
# Load Data
# =========================
df = pd.read_csv(RAW_DATA_PATH)

TARGET = "FraudResult"

# =========================
# 1️⃣ Aggregate Features
# =========================
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col="CustomerId", amount_col="Amount"):
        self.group_col = group_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = (
            X.groupby(self.group_col)
            .agg(
                TotalAmount=(self.amount_col, "sum"),
                AvgAmount=(self.amount_col, "mean"),
                TransactionCount=("TransactionId", "count"),
                StdAmount=(self.amount_col, "std"),
            )
            .reset_index()
        )
        return X.merge(agg_df, on=self.group_col, how="left")

# =========================
# 2️⃣ Time Features
# =========================
class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col="TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])

        X["TransactionHour"] = X[self.time_col].dt.hour
        X["TransactionDay"] = X[self.time_col].dt.day
        X["TransactionMonth"] = X[self.time_col].dt.month
        X["TransactionYear"] = X[self.time_col].dt.year
        X["TransactionWeekday"] = X[self.time_col].dt.weekday

        return X

# =========================
# 3️⃣ Separate X and y
# =========================
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# 4️⃣ Column Definitions
# =========================
numeric_cols = [
    "Amount",
    "Value",
    "TotalAmount",
    "AvgAmount",
    "StdAmount",
]

categorical_cols = [
    "ProductCategory",
    "ProviderId",
    "ChannelId",
]

id_cols = ["CustomerId"]

# =========================
# 5️⃣ Transformers
# =========================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("id", "passthrough", id_cols),
    ],
    remainder="drop",
)

# =========================
# 6️⃣ Full Pipeline
# =========================
pipeline = Pipeline(
    steps=[
        ("aggregate", AggregateFeatures()),
        ("time_features", TimeFeatures()),
        ("preprocessor", preprocessor),
    ]
)

# =========================
# 7️⃣ Apply Pipeline
# =========================
X_processed = pipeline.fit_transform(X)

# =========================
# 8️⃣ Build Output DataFrame
# =========================
cat_feature_names = (
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_cols)
)

final_columns = numeric_cols + list(cat_feature_names) + id_cols

df_processed = pd.DataFrame(X_processed, columns=final_columns)
df_processed[TARGET] = y.values

# =========================
# 9️⃣ Save
# =========================
df_processed.to_csv(PROCESSED_PATH, index=False)

print(f"✅ Processed dataset saved at: {PROCESSED_PATH}")
print(df_processed.head())
