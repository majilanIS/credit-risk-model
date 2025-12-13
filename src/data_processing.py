# =========================
# feature_engineering
# =========================

import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE

# =========================
# File Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/data.csv')
PROCESSED_PATH = os.path.join(BASE_DIR, '../data/processed/processed_transactions.csv')
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# =========================
# Load Raw Data
# =========================
df = pd.read_csv(RAW_DATA_PATH)
TARGET = 'FraudResult'  # target column

# =========================
# 1. Aggregate Features
# =========================
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', amount_col='Amount'):
        self.group_col = group_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.group_col).agg(
            TotalAmount=(self.amount_col, 'sum'),
            AvgAmount=(self.amount_col, 'mean'),
            TransactionCount=('TransactionId', 'count'),
            StdAmount=(self.amount_col, 'std')
        ).reset_index()
        X = X.merge(agg_df, on=self.group_col, how='left')
        return X

# =========================
# 2. Time Features
# =========================
class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        X['TransactionHour'] = X[self.time_col].dt.hour
        X['TransactionDay'] = X[self.time_col].dt.day
        X['TransactionMonth'] = X[self.time_col].dt.month
        X['TransactionYear'] = X[self.time_col].dt.year
        X['TransactionWeekday'] = X[self.time_col].dt.weekday
        return X

# =========================
# 3. Define Columns
# =========================
numeric_cols = ['Amount', 'Value', 'TotalAmount', 'AvgAmount', 'StdAmount']
categorical_cols = ['ProductCategory', 'ProviderId', 'ChannelId']
woe_cols = ['ChannelId']

# =========================
# 4. Transformers
# =========================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

woe_transformer = WOE(target=TARGET, variables=woe_cols)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
    ('woe', woe_transformer, woe_cols),
    ('id', 'passthrough', ['CustomerId'])
], remainder='drop')

# =========================
# 5. Full Pipeline
# =========================
full_pipeline = Pipeline([
    ('aggregate', AggregateFeatures()),
    ('time_features', TimeFeatures()),
    ('preprocessor', preprocessor)
])

# =========================
# 6. Apply Pipeline
# =========================
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_processed = full_pipeline.fit_transform(pd.concat([X, y], axis=1))

# =========================
# 7. Save Processed Dataset
# =========================
# Combine column names
num_cols = numeric_cols
cat_cols = list(full_pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot'].get_feature_names_out(categorical_cols))
woe_cols_names = woe_cols
final_columns = num_cols + cat_cols + woe_cols_names + ['CustomerId']

df_processed = pd.DataFrame(X_processed, columns=final_columns)
df_processed[TARGET] = y.values

df_processed.to_csv(PROCESSED_PATH, index=False)
print(f"Processed dataset saved at: {PROCESSED_PATH}")
print(df_processed.head())
