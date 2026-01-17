"""Batch prediction service for confirmed orders.

This module retrieves confirmed orders from the database, performs predictions,
and stores the results in a SQLite predictions table.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from delivery_ml.config import settings
from delivery_ml.features.store import FeatureStore
from delivery_ml.training.pipeline import DeliveryTimeModel


class BatchPredictor:
    """Batch prediction service for processing confirmed orders."""

    def __init__(
        self,
        model_path: Path | None = None,
        source_db_path: Path | None = None,
        predictions_db_path: Path | None = None,
    ):
        """Initialize the batch predictor.

        Args:
            model_path: Path to the trained model. Defaults to latest model.
            source_db_path: Path to the source SQLite database (orders, stores). Defaults to settings.sqlite_db_path.
            predictions_db_path: Path to store predictions. Defaults to 'predictions.db' in current directory.
        """
        self.source_db_path = source_db_path or settings.sqlite_db_path
        self.predictions_db_path = predictions_db_path or Path("predictions.db")
        self.model_path = model_path or settings.model_dir / "delivery_time_model_latest.pkl"
        
        print(f"Source database: {self.source_db_path}")
        print(f"Predictions database: {self.predictions_db_path}")
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = DeliveryTimeModel.load(self.model_path)
        print(f"Loaded model version: {self.model.version}")
        
        # Initialize feature store
        self.feature_store = FeatureStore()
        self.feature_store.initialize()
        print("Feature store initialized")
        
        # Initialize predictions table
        self._initialize_predictions_table()

    def _initialize_predictions_table(self) -> None:
        """Create predictions table if it doesn't exist."""
        conn = sqlite3.connect(self.predictions_db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    predicted_delivery_minutes REAL NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(order_id, model_version)
                )
            """)
            
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_order_id 
                ON ml_predictions(order_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                ON ml_predictions(prediction_timestamp)
            """)
            
            conn.commit()
            print("Predictions table initialized")
        finally:
            conn.close()

    def get_confirmed_orders(self, limit: int | None = None) -> pl.DataFrame:
        """Retrieve confirmed orders that don't have delivered_at set yet.

        Args:
            limit: Maximum number of orders to retrieve. None for all.

        Returns:
            DataFrame with confirmed orders.
        """
        conn = sqlite3.connect(self.source_db_path)
        try:
            # Join with stores to get store location and aggregate order_items for quantity
            query = """
                SELECT 
                    o.order_id,
                    o.customer_id,
                    o.store_id,
                    o.created_at,
                    s.latitude,
                    s.longitude,
                    o.delivery_latitude,
                    o.delivery_longitude,
                    o.total,
                    COALESCE(
                        (SELECT SUM(quantity) FROM order_items WHERE order_id = o.order_id),
                        1
                    ) as quantity
                FROM orders o
                LEFT JOIN stores s ON o.store_id = s.store_id
                WHERE o.delivered_at IS NULL
                  AND o.created_at IS NOT NULL
                  AND o.confirmed_at IS NOT NULL
                  AND s.latitude IS NOT NULL
                  AND s.longitude IS NOT NULL
                ORDER BY o.created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pl.read_database(query, connection=conn)
            print(f"Retrieved {len(df)} confirmed orders")
            return df
        finally:
            conn.close()

    def predict_for_order(self, order_row: dict[str, Any]) -> dict[str, Any]:
        """Generate prediction for a single order.

        Args:
            order_row: Dictionary containing order attributes.

        Returns:
            Dictionary with prediction results.
        """
        # Parse timestamp
        placed_at = datetime.fromisoformat(order_row["created_at"])
        
        # Get features from feature store
        features = self.feature_store.get_features_for_inference(
            customer_id=order_row["customer_id"],
            store_id=order_row["store_id"],
            restaurant_lat=order_row["latitude"],
            restaurant_lon=order_row["longitude"],
            delivery_lat=order_row["delivery_latitude"],
            delivery_lon=order_row["delivery_longitude"],
            placed_at=placed_at,
            order_total_cents=order_row["total"],
            item_count=order_row["quantity"],
        )
        
        # Make prediction
        prediction = self.model.predict(features)
        
        return {
            "order_id": order_row["order_id"],
            "customer_id": order_row["customer_id"],
            "store_id": order_row["store_id"],
            "predicted_delivery_minutes": prediction,
            "prediction_timestamp": placed_at.isoformat(),
            "model_version": self.model.version,
            "features": features,
        }

    def save_prediction(self, prediction: dict[str, Any]) -> None:
        """Save prediction to database.

        Args:
            prediction: Dictionary containing prediction results.
        """
        import json
        
        conn = sqlite3.connect(self.predictions_db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO ml_predictions (
                    order_id,
                    customer_id,
                    store_id,
                    predicted_delivery_minutes,
                    prediction_timestamp,
                    model_version,
                    features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction["order_id"],
                    prediction["customer_id"],
                    prediction["store_id"],
                    prediction["predicted_delivery_minutes"],
                    prediction["prediction_timestamp"],
                    prediction["model_version"],
                    json.dumps(prediction["features"]),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def run_batch_predictions(
        self,
        limit: int | None = None,
        save_to_db: bool = True,
    ) -> pl.DataFrame:
        """Run predictions on all confirmed orders.

        Args:
            limit: Maximum number of orders to process. None for all.
            save_to_db: Whether to save predictions to database.

        Returns:
            DataFrame with predictions.
        """
        # Get confirmed orders
        orders_df = self.get_confirmed_orders(limit=limit)
        
        if len(orders_df) == 0:
            print("No confirmed orders found")
            return pl.DataFrame()
        
        # Process each order
        predictions = []
        for row in orders_df.iter_rows(named=True):
            try:
                prediction = self.predict_for_order(row)
                predictions.append(prediction)
                
                if save_to_db:
                    self.save_prediction(prediction)
                
                print(
                    f"✓ Order {prediction['order_id']}: "
                    f"{prediction['predicted_delivery_minutes']:.1f} minutes"
                )
            except Exception as e:
                print(f"✗ Error predicting for order {row['order_id']}: {e}")
                continue
        
        # Convert to DataFrame
        predictions_df = pl.DataFrame(predictions)
        print(f"\nProcessed {len(predictions)} orders successfully")
        
        return predictions_df

    def get_prediction_stats(self) -> dict[str, Any]:
        """Get statistics about stored predictions.

        Returns:
            Dictionary with prediction statistics.
        """
        conn = sqlite3.connect(self.predictions_db_path)
        try:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT order_id) as unique_orders,
                    AVG(predicted_delivery_minutes) as avg_prediction,
                    MIN(predicted_delivery_minutes) as min_prediction,
                    MAX(predicted_delivery_minutes) as max_prediction,
                    MIN(prediction_timestamp) as earliest_prediction,
                    MAX(prediction_timestamp) as latest_prediction
                FROM ml_predictions
            """)
            
            row = cursor.fetchone()
            return {
                "total_predictions": row[0],
                "unique_orders": row[1],
                "avg_prediction_minutes": row[2],
                "min_prediction_minutes": row[3],
                "max_prediction_minutes": row[4],
                "earliest_prediction": row[5],
                "latest_prediction": row[6],
            }
        finally:
            conn.close()

    def close(self) -> None:
        """Clean up resources."""
        if self.feature_store:
            self.feature_store.close()


def main() -> None:
    """CLI entry point for batch predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run batch predictions on confirmed orders"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of orders to process (default: all)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save predictions to database",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show prediction statistics only",
    )
    
    args = parser.parse_args()
    
    predictor = BatchPredictor()
    
    try:
        if args.stats:
            stats = predictor.get_prediction_stats()
            print("\n=== Prediction Statistics ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            print("Starting batch predictions...")
            predictions_df = predictor.run_batch_predictions(
                limit=args.limit,
                save_to_db=not args.no_save,
            )
            
            if len(predictions_df) > 0:
                print("\n=== Summary ===")
                print(predictions_df.describe())
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
