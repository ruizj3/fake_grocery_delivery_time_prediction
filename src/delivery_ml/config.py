"""Application settings and configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="DELIVERY_ML_", env_file=".env")

    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    # Feature store - uses your existing SQLite database
    sqlite_db_path: Path = Path("grocery_delivery.db")
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Training
    train_test_split_date: str = "2024-01-01"
    feature_window_days: int = 30

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_version: str = "latest"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "delivery_time_prediction"


settings = Settings()
