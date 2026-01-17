from datetime import datetime
from delivery_ml.training.pipeline import train_model
from delivery_ml.features.store import OfflineFeatureStore

store = OfflineFeatureStore()
min_date, max_date = store.get_date_range()
print(f"Data range: {min_date} to {max_date}")

model_version = train_model(
    train_start=min_date,
    train_end=max_date,
)
print(f"Trained model: {model_version}")