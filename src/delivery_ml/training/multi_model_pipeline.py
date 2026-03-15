"""Multi-model training pipeline for delivery time prediction.

Supports multiple model types beyond XGBoost:
- LightGBM (recommended)
- Random Forest
- CatBoost
- Ridge Regression
- Gradient Boosting
- Extra Trees

All models support hyperparameter tuning via GridSearchCV or RandomizedSearchCV.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import mlflow
import polars as pl
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import randint, uniform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from delivery_ml.config import settings
from delivery_ml.features.definitions import TRAINING_FEATURES
from delivery_ml.features.store import FeatureStore

# Optional imports for advanced models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


if HAS_CATBOOST:
    class _SklearnCatBoostRegressor(BaseEstimator, RegressorMixin):
        """Sklearn-compatible wrapper for CatBoostRegressor.

        CatBoost's native sklearn wrapper lacks __sklearn_tags__ required by
        scikit-learn >= 1.6, which breaks GridSearchCV/RandomizedSearchCV.
        This thin wrapper inherits from BaseEstimator to provide compatibility.
        """

        def __init__(
            self,
            iterations: int = 500,
            depth: int = 10,
            learning_rate: float = 0.05,
            l2_leaf_reg: float = 3.0,
            loss_function: str = "RMSE",
            random_state: int = 42,
            verbose: bool = False,
            thread_count: int = -1,
        ):
            self.iterations = iterations
            self.depth = depth
            self.learning_rate = learning_rate
            self.l2_leaf_reg = l2_leaf_reg
            self.loss_function = loss_function
            self.random_state = random_state
            self.verbose = verbose
            self.thread_count = thread_count

        def fit(self, X, y, **fit_params):
            self.model_ = cb.CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                loss_function=self.loss_function,
                random_state=self.random_state,
                verbose=self.verbose,
                thread_count=self.thread_count,
            )
            self.model_.fit(X, y, **fit_params)
            return self

        def predict(self, X):
            return self.model_.predict(X)

        @property
        def feature_importances_(self):
            return self.model_.feature_importances_


ModelType = Literal["lightgbm", "random_forest", "catboost", "ridge", "gradient_boosting", "extra_trees"]


class MultiModelTrainer:
    """Training class supporting multiple model types."""

    def __init__(self, model_type: ModelType = "lightgbm"):
        self.model_type = model_type
        self.model: Any = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = TRAINING_FEATURES
        self.version: str = ""
        self.metrics: dict[str, float] = {}
        
        # Validate model availability
        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        if model_type == "catboost" and not HAS_CATBOOST:
            raise ImportError("CatBoost not installed. Run: pip install catboost")

    def _get_model(self, random_state: int = 42, **params: Any) -> Any:
        """Get model instance based on type."""
        if self.model_type == "lightgbm":
            default_params = {
                "n_estimators": 500,
                "max_depth": 15,
                "learning_rate": 0.05,
                "num_leaves": 50,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "regression",
                "metric": "rmse",
                "random_state": random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
            default_params.update(params)
            return lgb.LGBMRegressor(**default_params)
        
        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "random_state": random_state,
                "n_jobs": -1,
            }
            default_params.update(params)
            return RandomForestRegressor(**default_params)
        
        elif self.model_type == "catboost":
            default_params = {
                "iterations": 500,
                "depth": 10,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3,
                "loss_function": "RMSE",
                "random_state": random_state,
                "verbose": False,
                "thread_count": -1,
            }
            default_params.update(params)
            return cb.CatBoostRegressor(**default_params)
        
        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "subsample": 0.8,
                "random_state": random_state,
            }
            default_params.update(params)
            return GradientBoostingRegressor(**default_params)
        
        elif self.model_type == "extra_trees":
            default_params = {
                "n_estimators": 300,
                "max_depth": 25,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "random_state": random_state,
                "n_jobs": -1,
            }
            default_params.update(params)
            return ExtraTreesRegressor(**default_params)
        
        elif self.model_type == "ridge":
            default_params = {
                "alpha": 1.0,
                "random_state": random_state,
            }
            default_params.update(params)
            return Ridge(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_param_grid(self, use_distributions: bool = False) -> dict[str, Any]:
        """Get hyperparameter search space for the current model type.

        Args:
            use_distributions: If True, return scipy distributions for
                RandomizedSearchCV. If False, return discrete lists for GridSearchCV.
        """
        if self.model_type == "lightgbm":
            if use_distributions and HAS_SCIPY:
                return {
                    "n_estimators": randint(200, 800),
                    "max_depth": randint(5, 20),
                    "learning_rate": uniform(0.01, 0.19),
                    "num_leaves": randint(20, 80),
                    "min_child_samples": randint(10, 50),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                }
            return {
                "n_estimators": [300, 500, 700],
                "max_depth": [8, 12, 15],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 70],
                "min_child_samples": [10, 20, 30],
            }

        elif self.model_type == "random_forest":
            if use_distributions and HAS_SCIPY:
                return {
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(10, 30),
                    "min_samples_split": randint(5, 20),
                    "min_samples_leaf": randint(2, 10),
                }
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 4, 8],
                "max_features": ["sqrt", "log2"],
            }

        elif self.model_type == "catboost":
            if use_distributions and HAS_SCIPY:
                return {
                    "iterations": randint(300, 800),
                    "depth": randint(4, 12),
                    "learning_rate": uniform(0.01, 0.19),
                    "l2_leaf_reg": uniform(1, 9),
                }
            return {
                "iterations": [300, 500, 700],
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5, 7],
            }

        elif self.model_type == "gradient_boosting":
            if use_distributions and HAS_SCIPY:
                return {
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(4, 12),
                    "learning_rate": uniform(0.01, 0.19),
                    "min_samples_split": randint(5, 20),
                    "min_samples_leaf": randint(2, 10),
                    "subsample": uniform(0.6, 0.4),
                }
            return {
                "n_estimators": [200, 300, 400],
                "max_depth": [5, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9],
            }

        elif self.model_type == "extra_trees":
            if use_distributions and HAS_SCIPY:
                return {
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(10, 35),
                    "min_samples_split": randint(2, 15),
                    "min_samples_leaf": randint(1, 8),
                }
            return {
                "n_estimators": [200, 300, 400],
                "max_depth": [15, 25, 35],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            }

        elif self.model_type == "ridge":
            if use_distributions and HAS_SCIPY:
                return {
                    "alpha": uniform(0.01, 99.99),
                }
            return {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            }

        else:
            return {}

    def tune_hyperparameters(
        self,
        features_df: pl.DataFrame,
        target_col: str = "delivery_time_minutes",
        search_type: Literal["grid", "random"] = "random",
        cv: int = 5,
        n_iter: int = 30,
        scoring: str = "neg_mean_absolute_error",
        random_state: int = 42,
        param_grid: dict[str, Any] | None = None,
        n_jobs: int = -1,
    ) -> dict[str, Any]:
        """Tune hyperparameters using grid or randomized search.

        Args:
            features_df: Training data with features and target.
            target_col: Name of the target column.
            search_type: "random" (faster, recommended) or "grid" (exhaustive).
            cv: Number of cross-validation folds.
            n_iter: Number of parameter combinations for randomized search.
            scoring: Scoring metric (sklearn convention, e.g. "neg_mean_absolute_error").
            random_state: Random seed for reproducibility.
            param_grid: Custom parameter grid. If None, uses built-in defaults.
            n_jobs: Number of parallel jobs (-1 = all cores).

        Returns:
            Dictionary with best_params, best_score, and cv_results summary.
        """
        # Prepare data
        X = features_df.select(self.feature_names).to_pandas().fillna(-1)
        y = features_df.select(target_col).to_pandas().values.ravel()

        # Scale for Ridge
        if self.model_type == "ridge":
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Get base estimator (use sklearn-compatible wrapper for CatBoost)
        if self.model_type == "catboost" and HAS_CATBOOST:
            base_model = _SklearnCatBoostRegressor(random_state=random_state)
        else:
            base_model = self._get_model(random_state=random_state)

        # Get parameter grid
        if param_grid is None:
            use_distributions = search_type == "random"
            param_grid = self._get_param_grid(use_distributions=use_distributions)

        if not param_grid:
            raise ValueError(f"No parameter grid defined for model type: {self.model_type}")

        # Run search
        if search_type == "random":
            print(f"Running RandomizedSearchCV for {self.model_type} "
                  f"({n_iter} iterations, {cv}-fold CV)...")
            searcher = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=1,
                refit=False,
            )
        else:
            # Calculate total combinations for user feedback
            from itertools import product as _product
            total = 1
            for v in param_grid.values():
                total *= len(v)
            print(f"Running GridSearchCV for {self.model_type} "
                  f"({total} combinations, {cv}-fold CV)...")
            searcher = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                refit=False,
            )

        searcher.fit(X, y)

        best_params = searcher.best_params_
        best_score = -searcher.best_score_  # negate back since sklearn uses neg metrics

        print(f"\n✅ Best {scoring.replace('neg_', '')}: {best_score:.4f}")
        print(f"Best parameters:")
        for param, value in sorted(best_params.items()):
            print(f"  {param}: {value}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "scoring": scoring,
            "search_type": search_type,
            "cv_folds": cv,
        }

    def train(
        self,
        features_df: pl.DataFrame,
        target_col: str = "delivery_time_minutes",
        test_size: float | None = None,
        random_state: int = 42,
        tune: bool = False,
        tune_kwargs: dict[str, Any] | None = None,
        **model_params: Any,
    ) -> dict[str, float]:
        """Train the model on features DataFrame.

        Args:
            features_df: Training data with features and target.
            target_col: Name of the target column.
            test_size: Fraction of data to hold out for testing.
            random_state: Random seed for reproducibility.
            tune: If True, run hyperparameter tuning before training.
            tune_kwargs: Additional keyword arguments for tune_hyperparameters.
            **model_params: Override default model parameters.
        """
        # Optionally tune hyperparameters first
        if tune:
            print("\n--- Hyperparameter Tuning ---")
            tk = tune_kwargs or {}
            tk.setdefault("random_state", random_state)
            tune_results = self.tune_hyperparameters(features_df, target_col, **tk)
            # Merge tuned params (explicit model_params take priority)
            tuned_params = tune_results["best_params"]
            tuned_params.update(model_params)
            model_params = tuned_params
            # Reset scaler so train split gets its own fit
            self.scaler = None
            print("\n--- Training with Tuned Parameters ---")

        # Prepare data
        X = features_df.select(self.feature_names).to_pandas()
        y = features_df.select(target_col).to_pandas().values.ravel()

        # Handle missing values
        X = X.fillna(-1)

        # Use config test_size if not provided
        if test_size is None:
            test_size = settings.test_size

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features for Ridge regression
        if self.model_type == "ridge":
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # Get and train model
        self.model = self._get_model(random_state=random_state, **model_params)
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred_test),
            "rmse": mean_squared_error(y_test, y_pred_test) ** 0.5,
            "r2": r2_score(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
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

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return float(self.model.predict(X)[0])

    def predict_batch(self, features_df: pl.DataFrame) -> list[float]:
        """Predict delivery time for multiple orders."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        X = features_df.select(self.feature_names).to_pandas().fillna(-1)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X).tolist()

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "model_type": self.model_type,
                    "feature_names": self.feature_names,
                    "version": self.version,
                    "metrics": self.metrics,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "MultiModelTrainer":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.scaler = data.get("scaler")
        instance.feature_names = data["feature_names"]
        instance.version = data["version"]
        instance.metrics = data["metrics"]
        return instance

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores (if available)."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        # Different models have different ways to get importance
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        elif self.model_type == "ridge" and hasattr(self.model, "coef_"):
            # For Ridge, use absolute coefficient values
            importance = abs(self.model.coef_)
            return dict(zip(self.feature_names, importance.tolist()))
        else:
            return {}


class MultiModelPipeline:
    """Training pipeline with multiple model support."""

    def __init__(
        self,
        model_type: ModelType = "lightgbm",
        feature_store: FeatureStore | None = None,
    ):
        self.model_type = model_type
        self.feature_store = feature_store or FeatureStore()
        self.model = MultiModelTrainer(model_type)

    def run(
        self,
        train_start: datetime,
        train_end: datetime,
        experiment_name: str | None = None,
        model_name: str = "delivery_time_model",
        **model_params: Any,
    ) -> str:
        """Run the full training pipeline with MLflow tracking."""
        experiment_name = experiment_name or settings.mlflow_experiment_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{self.model_type}_model") as run:
            # Log parameters
            mlflow.log_params(
                {
                    "model_type": self.model_type,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "features": ",".join(TRAINING_FEATURES),
                    **model_params,
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
            metrics = self.model.train(features_df, **model_params)

            # Log metrics
            mlflow.log_metrics(metrics)
            print(f"\nMetrics:")
            print(f"  Test  - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
            print(f"  Train - MAE: {metrics['train_mae']:.2f}, R²: {metrics['train_r2']:.3f}")

            # Log feature importance
            importance = self.model.get_feature_importance()
            if importance:
                print(f"\nTop 5 Most Important Features:")
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for name, score in sorted_importance[:5]:
                    mlflow.log_metric(f"importance_{name}", score)
                    print(f"  {name}: {score:.4f}")

            # Save model
            self.model.version = run.info.run_id
            model_path = settings.model_dir / f"{model_name}_{self.model_type}_{run.info.run_id}.pkl"
            self.model.save(model_path)

            # Log model artifact
            mlflow.log_artifact(str(model_path))

            # Also save as "latest" for this model type
            latest_path = settings.model_dir / f"{model_name}_{self.model_type}_latest.pkl"
            self.model.save(latest_path)

            print(f"\nModel saved: {model_path}")
            print(f"MLflow run ID: {run.info.run_id}")

            return run.info.run_id


def train_model(
    model_type: ModelType = "lightgbm",
    train_start: datetime | None = None,
    train_end: datetime | None = None,
    materialize_features: bool = True,
    **kwargs: Any,
) -> str:
    """
    Convenience function to train a model with specified type.
    
    Args:
        model_type: Type of model to train
            (lightgbm, random_forest, catboost, ridge, gradient_boosting, extra_trees)
        train_start: Start date for training data
        train_end: End date for training data
        materialize_features: If True, materializes aggregated features before training
        **kwargs: Additional parameters forwarded to model.train(), including:
            - tune (bool): Enable hyperparameter tuning before training
            - tune_kwargs (dict): Options for tuning (search_type, cv, n_iter, etc.)
    """
    pipeline = MultiModelPipeline(model_type=model_type)
    pipeline.feature_store.initialize()
    
    # If dates not provided, use last 31 days
    if train_start is None or train_end is None:
        min_date, max_date = pipeline.feature_store.offline.get_date_range()
        train_end = max_date
        train_start = max_date - timedelta(days=31)
        print(f"Using last 31 days of data: {train_start} to {train_end}")
    
    # Materialize aggregated features for the training window
    if materialize_features:
        print("\nMaterializing aggregated features for training window...")
        days_in_window = (train_end - train_start).days
        materialized_count = 0
        
        for days_back in range(days_in_window):
            as_of = train_end - timedelta(days=days_back)
            try:
                count = pipeline.feature_store.offline.materialize_restaurant_features(
                    as_of, window_days=30
                )
                materialized_count += count
                
                if days_back > 0 and days_back % 7 == 0:
                    print(f"  Progress: {days_back}/{days_in_window} days processed...")
            except Exception as e:
                print(f"  Warning: Could not materialize features for {as_of.date()}: {e}")
        
        print(f"✅ Materialized features for {materialized_count} restaurants")
    
    return pipeline.run(train_start, train_end, **kwargs)
