# =========================
# feature_engineering
# =========================

import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearo.encoding import WOEEncoder 

# =========================
# 0. File Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/data.csv')
PROCESSED_PATH = os.path.join(BASE_DIR, '../data/processed/processed_transactions.csv')
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# =========================
# 1. Load Raw Data
# =========================
df = pd.read_csv(RAW_DATA_PATH)

# =========================
# 2. Custom Transformers
# =========================

# 2.1 Aggregate Features
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Compute per-customer aggregated features:
    TotalAmount, AvgAmount, TransactionCount, StdAmount
    """
    def __init__(self, group_col='CustomerId', amount_col='Amount'):
        self.group_col = group_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 2.1.1 Group by customer and compute aggregates
        agg_df = X.groupby(self.group_col).agg(
            TotalAmount=(self.amount_col, 'sum'),
            AvgAmount=(self.amount_col, 'mean'),
            TransactionCount=('TransactionId', 'count'),
            StdAmount=(self.amount_col, 'std')
        ).reset_index()
        # 2.1.2 Merge aggregated features back to main df
        X = X.merge(agg_df, on=self.group_col, how='left')
        return X

# 2.2 Time Features
class TimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract features from TransactionStartTime:
    Hour, Day, Month, Year, Weekday
    """
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 2.2.1 Convert to datetime
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        # 2.2.2 Extract time components
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
# 4. Build Transformers
# =========================

# 4.1 Numeric Pipeline: Handle missing and standardize
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())                 
])

# 4.2 Categorical Pipeline: Handle missing and one-hot encode
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')) 
])

# 4.3 WoE Transformer 
# woe_transformer = WOEEncoder() 

# 4.4 ColumnTransformer: Apply pipelines
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
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
df_features = full_pipeline.fit_transform(df)

# 6.1 Combine processed features and CustomerId
all_columns = (
    numeric_cols +
    list(full_pipeline.named_steps['preprocessor'].named_transformers_['cat']
         .named_steps['onehot'].get_feature_names_out(categorical_cols)) +
    # woe_cols +  # Uncomment if WoE used
    ['CustomerId']
)

df_processed = pd.DataFrame(df_features, columns=all_columns)

# =========================
# 7. Save Processed Dataset
# =========================
df_processed.to_csv(PROCESSED_PATH, index=False)
print(f"Processed dataset saved at: {PROCESSED_PATH}")
print(df_processed.head())
