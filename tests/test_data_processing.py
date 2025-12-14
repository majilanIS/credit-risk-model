# tests/test_data_processing.py

import os
import pandas as pd
import pytest

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, "../data/processed/processed_transactions.csv")

# =========================
# Fixture: Load processed data
# =========================
@pytest.fixture
def processed_data():
    df = pd.read_csv(PROCESSED_PATH)
    return df

# =========================
# Test 1: Check required columns exist
# =========================
def test_columns_exist(processed_data):
    """
    Check that essential columns exist in the processed dataset.
    Works with one-hot encoded / WOE features using prefix matching.
    """
    required_prefixes = [
        "Amount", "Value", "TotalAmount", "AvgAmount", "StdAmount",
        "ProductCategory", "ProviderId", "ChannelId", "CustomerId", "FraudResult"
    ]
    
    for prefix in required_prefixes:
        assert any(col.startswith(prefix) for col in processed_data.columns), \
            f"Column starting with '{prefix}' not found in processed data"

# =========================
# Test 2: Check target column has no missing values
# =========================
def test_target_no_missing(processed_data):
    """
    Ensure that the target column 'FraudResult' has no missing values.
    """
    assert processed_data["FraudResult"].isnull().sum() == 0, \
        "Target column 'FraudResult' contains missing values"