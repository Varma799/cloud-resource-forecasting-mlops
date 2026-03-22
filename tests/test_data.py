import os
import pandas as pd


def test_ingested_file_exists():
    assert os.path.exists("data/processed/ingested.csv")


def test_transformed_file_exists():
    assert os.path.exists("data/processed/transformed.csv")


def test_transformed_data_has_required_columns():
    df = pd.read_csv("data/processed/transformed.csv")

    required_columns = [
        "cpu_usage",
        "memory_usage",
        "hour",
        "day_of_week",
        "cpu_lag_1",
        "memory_lag_1",
        "future_cpu_usage",
        "future_memory_usage",
        "scaling_action"
    ]

    for col in required_columns:
        assert col in df.columns