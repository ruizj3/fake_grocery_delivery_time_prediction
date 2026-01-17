"""Drift detection and monitoring for delivery time prediction model.

Monitors for:
1. Feature drift: input distributions shifting from training data
2. Prediction drift: model outputs shifting over time
3. Concept drift: relationship between features and target changing
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl
from scipy import stats

from delivery_ml.features.definitions import TRAINING_FEATURES


@dataclass
class DriftResult:
    """Result of a drift detection test."""

    feature_name: str
    test_name: str
    statistic: float
    p_value: float
    is_drifted: bool
    threshold: float
    reference_mean: float | None = None
    current_mean: float | None = None


class DriftDetector:
    """Detects drift between reference and current data distributions."""

    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold
        self.reference_stats: dict[str, dict[str, float]] = {}

    def fit_reference(self, reference_df: pl.DataFrame) -> None:
        """Compute reference statistics from training data."""
        for feature in TRAINING_FEATURES:
            if feature not in reference_df.columns:
                continue

            col = reference_df[feature].drop_nulls()

            if col.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                self.reference_stats[feature] = {
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "median": float(col.median()),
                }

    def detect_drift_ks_test(
        self,
        reference_values: list[float],
        current_values: list[float],
        feature_name: str,
    ) -> DriftResult:
        """
        Detect drift using Kolmogorov-Smirnov test.

        Good for detecting any difference in distributions.
        """
        if len(reference_values) < 10 or len(current_values) < 10:
            return DriftResult(
                feature_name=feature_name,
                test_name="ks_test",
                statistic=0.0,
                p_value=1.0,
                is_drifted=False,
                threshold=self.p_value_threshold,
                reference_mean=None,
                current_mean=None,
            )

        statistic, p_value = stats.ks_2samp(reference_values, current_values)

        return DriftResult(
            feature_name=feature_name,
            test_name="ks_test",
            statistic=statistic,
            p_value=p_value,
            is_drifted=p_value < self.p_value_threshold,
            threshold=self.p_value_threshold,
            reference_mean=sum(reference_values) / len(reference_values),
            current_mean=sum(current_values) / len(current_values),
        )

    def detect_drift_psi(
        self,
        reference_values: list[float],
        current_values: list[float],
        feature_name: str,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
    ) -> DriftResult:
        """
        Detect drift using Population Stability Index (PSI).

        PSI < 0.1: no significant change
        PSI 0.1-0.2: moderate change
        PSI > 0.2: significant change
        """
        if len(reference_values) < 10 or len(current_values) < 10:
            return DriftResult(
                feature_name=feature_name,
                test_name="psi",
                statistic=0.0,
                p_value=1.0,
                is_drifted=False,
                threshold=psi_threshold,
            )

        # Create bins from reference distribution
        min_val = min(min(reference_values), min(current_values))
        max_val = max(max(reference_values), max(current_values))
        bins = [min_val + i * (max_val - min_val) / n_bins for i in range(n_bins + 1)]

        # Calculate proportions
        ref_counts, _ = pl.Series(reference_values).hist(bins=bins)
        cur_counts, _ = pl.Series(current_values).hist(bins=bins)

        # Normalize to proportions
        ref_props = [c / len(reference_values) for c in ref_counts]
        cur_props = [c / len(current_values) for c in cur_counts]

        # Calculate PSI
        psi = 0.0
        for ref_p, cur_p in zip(ref_props, cur_props):
            # Avoid division by zero
            ref_p = max(ref_p, 0.0001)
            cur_p = max(cur_p, 0.0001)
            psi += (cur_p - ref_p) * (cur_p / ref_p if ref_p > 0 else 0)

        return DriftResult(
            feature_name=feature_name,
            test_name="psi",
            statistic=psi,
            p_value=1 - min(psi / psi_threshold, 1.0),  # Pseudo p-value
            is_drifted=psi > psi_threshold,
            threshold=psi_threshold,
            reference_mean=sum(reference_values) / len(reference_values),
            current_mean=sum(current_values) / len(current_values),
        )

    def check_all_features(
        self,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
    ) -> list[DriftResult]:
        """Check drift for all features."""
        results = []

        for feature in TRAINING_FEATURES:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue

            ref_values = reference_df[feature].drop_nulls().to_list()
            cur_values = current_df[feature].drop_nulls().to_list()

            # Run KS test
            ks_result = self.detect_drift_ks_test(ref_values, cur_values, feature)
            results.append(ks_result)

            # Run PSI
            psi_result = self.detect_drift_psi(ref_values, cur_values, feature)
            results.append(psi_result)

        return results

    def generate_report(self, results: list[DriftResult]) -> dict[str, Any]:
        """Generate a drift detection report."""
        drifted_features = [r for r in results if r.is_drifted]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": len(results),
            "drifted_count": len(drifted_features),
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": [
                {
                    "feature": r.feature_name,
                    "test": r.test_name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "reference_mean": r.reference_mean,
                    "current_mean": r.current_mean,
                }
                for r in drifted_features
            ],
            "all_results": [
                {
                    "feature": r.feature_name,
                    "test": r.test_name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "is_drifted": r.is_drifted,
                }
                for r in results
            ],
        }


class PredictionMonitor:
    """Monitors model predictions over time."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: list[dict[str, Any]] = []
        self.reference_predictions: list[float] = []

    def set_reference(self, predictions: list[float]) -> None:
        """Set reference predictions from training or validation data."""
        self.reference_predictions = predictions

    def log_prediction(
        self,
        order_id: str,
        prediction: float,
        features: dict[str, Any],
        actual: float | None = None,
    ) -> None:
        """Log a prediction for monitoring."""
        self.predictions.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "order_id": order_id,
                "prediction": prediction,
                "actual": actual,
                "features": features,
            }
        )

        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)

    def get_prediction_drift(self) -> DriftResult | None:
        """Check if predictions are drifting from reference."""
        if not self.reference_predictions or len(self.predictions) < 100:
            return None

        current_predictions = [p["prediction"] for p in self.predictions[-500:]]

        detector = DriftDetector()
        return detector.detect_drift_ks_test(
            self.reference_predictions,
            current_predictions,
            "predictions",
        )

    def get_error_stats(self) -> dict[str, float] | None:
        """Get error statistics for predictions with known actuals."""
        with_actuals = [p for p in self.predictions if p["actual"] is not None]

        if len(with_actuals) < 10:
            return None

        errors = [p["prediction"] - p["actual"] for p in with_actuals]
        abs_errors = [abs(e) for e in errors]

        return {
            "mae": sum(abs_errors) / len(abs_errors),
            "mean_error": sum(errors) / len(errors),  # Bias
            "n_samples": len(with_actuals),
        }
