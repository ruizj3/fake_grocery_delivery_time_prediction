#!/usr/bin/env python3
"""Train delivery time prediction model."""

from datetime import datetime
from delivery_ml.training.pipeline import train_model
from delivery_ml.features.store import OfflineFeatureStore


def main():
    """Train model using the last 31 days of data with 20% test split."""
    # train_model will automatically use last 31 days if no dates provided
    model_version = train_model()
    print(f"Trained model: {model_version}")


if __name__ == "__main__":
    main()
