# =========================
# Proxy Target Variable Engineering
# =========================

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# Step 0: File Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/data.csv')             
PROCESSED_PATH = os.path.join(BASE_DIR, '../data/processed/processed_transactions.csv')  

os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)

# =========================
# Step 1: Load Data
# =========================
df_raw = pd.read_csv(RAW_DATA_PATH)          
df_processed = pd.read_csv(PROCESSED_PATH)   

# Ensure TransactionStartTime is datetime
df_raw['TransactionStartTime'] = pd.to_datetime(df_raw['TransactionStartTime'])

# =========================
# Step 2: Calculate RFM Metrics
# =========================
snapshot_date = df_raw['TransactionStartTime'].max() + pd.Timedelta(days=1)

rfm = df_raw.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  
    'TransactionId': 'count',                                          
    'Amount': 'sum'                                                     
}).reset_index()

rfm.rename(columns={
    'TransactionStartTime': 'Recency',
    'TransactionId': 'Frequency',
    'Amount': 'Monetary'
}, inplace=True)

# =========================
# Step 3: Scale RFM Features
# =========================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# =========================
# Step 4: K-Means Clustering
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# =========================
# Step 5: Identify High-Risk Cluster
# =========================
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

# High-risk = highest Recency, lowest Frequency & Monetary
high_risk_cluster = cluster_summary.sort_values(
    ['Recency', 'Frequency', 'Monetary'],
    ascending=[False, True, True]
).index[0]

rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)

# =========================
# Step 6: Merge Target Variable into Processed Dataset
# =========================

# Step 6a: Ensure processed dataset has CustomerId
if 'CustomerId' not in df_processed.columns:
    df_processed['CustomerId'] = df_raw['CustomerId']

# Step 6b: Merge using CustomerId
df_final = df_processed.merge(
    rfm[['CustomerId', 'is_high_risk']],
    on='CustomerId',
    how='left'
)

# Fill any missing high-risk values with 0
df_final['is_high_risk'] = df_final['is_high_risk'].fillna(0).astype(int)

# =========================
# Step 7: Save Final Dataset
# =========================
df_final.to_csv(FINAL_OUTPUT_PATH, index=False)
print(f"Final dataset saved to: {FINAL_OUTPUT_PATH}")
print(df_final.head())
