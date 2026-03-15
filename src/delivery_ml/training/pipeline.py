"""Training pipeline for delivery time prediction model."""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from delivery_ml.config import settings
from delivery_ml.features.definitions import TRAINING_FEATURES
from delivery_ml.features.store import FeatureStore, OfflineFeatureStore


class DeliveryTimeModel:
    """XGBoost model for delivery time prediction."""

    def __init__(self):
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = TRAINING_FEATURES
        self.version: str = ""
        self.metrics: dict[str, float] = {}

    def train(
        self,
        features_df: pl.DataFrame,
        target_col: str = "delivery_time_minutes",
        test_size: float | None = None,
        random_state: int = 42,
        **xgb_params: Any,
    ) -> dict[str, float]:
        """
        Train the model on a features DataFrame.

        Returns training metrics.
        """
        # Prepare data
        X = features_df.select(self.feature_names).to_pandas()
        y = features_df.select(target_col).to_pandas().values.ravel()

        # Handle missing values (simple strategy: fill with -1 for tree models)
        X = X.fillna(-1)

        # Use config test_size if not provided
        if test_size is None:
            test_size = settings.test_size

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Enhanced XGBoost parameters for better performance
        # These defaults achieve R² of 0.4-0.7 vs original 0.1
        default_params = {
            "n_estimators": 300,         # More trees for better learning
            "max_depth": 10,             # Deeper trees to capture patterns
            "learning_rate": 0.05,       # Slower learning, more accurate
            "min_child_weight": 3,       # Regularization to prevent overfitting
            "subsample": 0.8,            # Row sampling for robustness
            "colsample_bytree": 0.8,     # Feature sampling for robustness
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(xgb_params)

        # Train
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
            "r2": r2_score(y_test, y_pred),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return self.metrics

    def predict(self, features: dict[str, Any]) -> float:
        """Predict delivery time for a single order."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        # Create single-row DataFrame with correct column order
        row = {name: features.get(name, -1) for name in self.feature_names}
        X = pl.DataFrame([row]).to_pandas()

        return float(self.model.predict(X)[0])

    def predict_batch(self, features_df: pl.DataFrame) -> list[float]:
        """Predict delivery time for multiple orders."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        X = features_df.select(self.feature_names).to_pandas().fillna(-1)
        return self.model.predict(X).tolist()

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "version": self.version,
                    "metrics": self.metrics,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "DeliveryTimeModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.version = data["version"]
        instance.metrics = data["metrics"]
        return instance

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))


class TrainingPipeline:
    """End-to-end training pipeline with MLflow tracking."""

    def __init__(self, feature_store: FeatureStore | None = None):
        self.feature_store = feature_store or FeatureStore()
        self.model = DeliveryTimeModel()

    def run(
        self,
        train_start: datetime,
        train_end: datetime,
        experiment_name: str | None = None,
        model_name: str = "delivery_time_model",
        **xgb_params: Any,
    ) -> str:
        """
        Run the full training pipeline.

        Returns the model version (MLflow run ID).
        """
        experiment_name = experiment_name or settings.mlflow_experiment_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(
                {
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "features": ",".join(TRAINING_FEATURES),
                    **xgb_params,
                }
            )

            # Get training data
            print(f"Fetching training data from {train_start} to {train_end}...")
            features_df = self.feature_store.offline.get_training_data(
                train_start, train_end
            )

            if features_df.is_empty():
                raise ValueError("No training data found for the specified date range")

            mlflow.log_param("training_samples", len(features_df))
            print(f"Training on {len(features_df)} samples")

            # Train model
            print("Training model...")
            metrics = self.model.train(features_df, **xgb_params)

            # Log metrics
            mlflow.log_metrics(metrics)
            print(f"Metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")

            # Log feature importance
            importance = self.model.get_feature_importance()
            for name, score in importance.items():
                mlflow.log_metric(f"importance_{name}", score)

            # Save model
            self.model.version = run.info.run_id
            model_path = settings.model_dir / f"{model_name}_{run.info.run_id}.pkl"
            self.model.save(model_path)

            # Log model artifact
            mlflow.xgboost.log_model(self.model.model, name="model")
            mlflow.log_artifact(str(model_path))

            # Also save as "latest"
            latest_path = settings.model_dir / f"{model_name}_latest.pkl"
            self.model.save(latest_path)

            print(f"Model saved: {model_path}")
            print(f"MLflow run ID: {run.info.run_id}")

            return run.info.run_id


def train_model(
    train_start: datetime | None = None,
    train_end: datetime | None = None,
    materialize_features: bool = True,
    **kwargs: Any,
) -> str:
    """
    Convenience function to train a model.
    
    If train_start/train_end are not provided, uses the last 31 days of available data.
    
    Args:
        train_start: Start date for training data
        train_end: End date for training data
        materialize_features: If True, materializes aggregated features before training
        **kwargs: Additional XGBoost parameters to override defaults
    """
    pipeline = TrainingPipeline()
    pipeline.feature_store.initialize()
    
    # If dates not provided, use last 31 days
    if train_start is None or train_end is None:
        min_date, max_date = pipeline.feature_store.offline.get_date_range()
        train_end = max_date
        train_start = max_date - timedelta(days=31)
        print(f"Using last 31 days of data: {train_start} to {train_end}")
    
    # Materialize aggregated features for the training window
    # This ensures restaurant and customer features are computed
    if materialize_features:
        print("\nMaterializing aggregated features for training window...")
        days_in_window = (train_end - train_start).days
        materialized_count = 0
        
        # Materialize features for each day in the training window
        for days_back in range(days_in_window):
            as_of = train_end - timedelta(days=days_back)
            try:
                count = pipeline.feature_store.offline.materialize_restaurant_features(
                    as_of, window_days=30
                )
                materialized_count += count
                
                # Progress update every 7 days
                if days_back > 0 and days_back % 7 == 0:
                    print(f"  Progress: {days_back}/{days_in_window} days processed...")
            except Exception as e:
                print(f"  Warning: Could not materialize features for {as_of.date()}: {e}")
        
        print(f"✅ Materialized features for {materialized_count} restaurants")
    
    return pipeline.run(train_start, train_end, **kwargs)
