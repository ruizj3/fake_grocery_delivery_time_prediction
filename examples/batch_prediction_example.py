"""
Example: Running Batch Predictions

This script demonstrates how to use the batch predictor to:
1. Retrieve confirmed orders from the database
2. Generate predictions for each order
3. Store predictions in the ml_predictions table
4. Query and analyze the results
"""

import polars as pl
import sqlite3
from delivery_ml.serving.batch_predictor import BatchPredictor
from delivery_ml.config import settings


def main():
    print("=== Batch Prediction Example ===\n")
    
    # Initialize the batch predictor
    print("1. Initializing batch predictor...")
    predictor = BatchPredictor()
    print(f"   Model version: {predictor.model.version}")
    print(f"   Database: {predictor.db_path}\n")
    
    # Check how many confirmed orders exist
    orders_df = predictor.get_confirmed_orders()
    print(f"2. Found {len(orders_df)} confirmed orders (without delivered_at)\n")
    
    if len(orders_df) == 0:
        print("   No confirmed orders to process. Exiting.")
        predictor.close()
        return
    
    # Show sample of confirmed orders
    print("   Sample orders:")
    print(orders_df.head(3).select(["order_id", "customer_id", "store_id", "created_at"]))
    print()
    
    # Run predictions on first 10 orders
    print("3. Running predictions on first 10 orders...")
    predictions_df = predictor.run_batch_predictions(limit=10, save_to_db=True)
    print()
    
    # Show prediction results
    if len(predictions_df) > 0:
        print("4. Prediction results:")
        print(predictions_df.select([
            "order_id",
            "predicted_delivery_minutes",
            "model_version"
        ]))
        print()
        
        # Get statistics
        print("5. Prediction statistics:")
        stats = predictor.get_prediction_stats()
        print(f"   Total predictions in database: {stats['total_predictions']}")
        print(f"   Unique orders predicted: {stats['unique_orders']}")
        print(f"   Average prediction: {stats['avg_prediction_minutes']:.2f} minutes")
        print(f"   Min prediction: {stats['min_prediction_minutes']:.2f} minutes")
        print(f"   Max prediction: {stats['max_prediction_minutes']:.2f} minutes")
        print()
        
        # Query predictions from database to verify they were saved
        print("6. Querying saved predictions from database...")
        conn = sqlite3.connect(predictor.db_path)
        saved_predictions = pl.read_database(
            """
            SELECT 
                order_id,
                predicted_delivery_minutes,
                model_version,
                prediction_timestamp
            FROM ml_predictions
            ORDER BY created_at DESC
            LIMIT 5
            """,
            connection=conn
        )
        conn.close()
        
        print("   Latest predictions in database:")
        print(saved_predictions)
        print()
        
    # Clean up
    predictor.close()
    
    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("- Run 'python batch_predict.py' to process all confirmed orders")
    print("- Run 'python batch_predict.py --stats' to see full statistics")
    print("- Query ml_predictions table to analyze results")


if __name__ == "__main__":
    main()
