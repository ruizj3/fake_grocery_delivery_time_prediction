"""
Prediction Accuracy Analysis

This script analyzes the accuracy of stored predictions by comparing them
to actual delivery times once orders are completed.
"""

import sqlite3
from pathlib import Path

import polars as pl

from delivery_ml.config import settings


def export_predictions_to_csv(
    predictions_db_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Export all tables from predictions database to CSV files.
    
    Args:
        predictions_db_path: Path to predictions database. Defaults to 'predictions.db'.
        output_dir: Output directory for CSV files. Defaults to 'exports/'.
    """
    predictions_db_path = predictions_db_path or Path("predictions.db")
    output_dir = output_dir or Path("exports")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    print("=== Exporting Predictions Database to CSV ===\n")
    print(f"Database: {predictions_db_path}")
    print(f"Output directory: {output_dir}\n")
    
    if not predictions_db_path.exists():
        print(f"Error: Database not found at {predictions_db_path}")
        print("Run: python batch_predict.py first")
        return
    
    conn = sqlite3.connect(predictions_db_path)
    
    try:
        # Get all table names
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print("No tables found in database")
            return
        
        print(f"Found {len(tables)} table(s) to export:\n")
        
        exported_count = 0
        for table_name in tables:
            try:
                # Read table data
                df = pl.read_database(
                    f"SELECT * FROM {table_name}",
                    connection=conn,
                )
                
                # Export to CSV
                csv_path = output_dir / f"{table_name}.csv"
                df.write_csv(csv_path)
                
                print(f"✓ {table_name}: {len(df)} rows → {csv_path}")
                exported_count += 1
                
            except Exception as e:
                print(f"✗ {table_name}: Error - {e}")
        
        print(f"\n✓ Successfully exported {exported_count}/{len(tables)} table(s)")
        
    finally:
        conn.close()


def analyze_prediction_accuracy(
    predictions_db_path: Path | None = None,
    source_db_path: Path | None = None,
) -> None:
    """Analyze prediction accuracy for completed orders.
    
    Args:
        predictions_db_path: Path to predictions database. Defaults to 'predictions.db'.
        source_db_path: Path to source database. Defaults to settings.sqlite_db_path.
    """
    predictions_db_path = predictions_db_path or Path("predictions.db")
    source_db_path = source_db_path or settings.sqlite_db_path
    
    print("=== Prediction Accuracy Analysis ===\n")
    print(f"Predictions DB: {predictions_db_path}")
    print(f"Source DB: {source_db_path}\n")
    
    # Read predictions from local database
    predictions_conn = sqlite3.connect(predictions_db_path)
    
    try:
        predictions_df = pl.read_database(
            """
            SELECT 
                order_id,
                customer_id,
                store_id,
                predicted_delivery_minutes,
                prediction_timestamp,
                model_version
            FROM ml_predictions
            ORDER BY prediction_timestamp DESC
            """,
            connection=predictions_conn,
        )
    except Exception as e:
        print(f"Error reading predictions: {e}")
        print("\nMake sure you have run: python batch_predict.py")
        predictions_conn.close()
        return
    
    predictions_conn.close()
    
    if len(predictions_df) == 0:
        print("No predictions found in database.")
        print("Run: python batch_predict.py")
        return
    
    # Read actual delivery times from source database
    source_conn = sqlite3.connect(source_db_path)
    
    try:
        actuals_df = pl.read_database(
            """
            SELECT 
                order_id,
                created_at,
                delivered_at,
                CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL) as actual_delivery_minutes
            FROM orders
            WHERE delivered_at IS NOT NULL
            """,
            connection=source_conn,
        )
    except Exception as e:
        print(f"Error reading source data: {e}")
        source_conn.close()
        return
    
    source_conn.close()
    
    # Join predictions with actuals
    df = predictions_df.join(actuals_df, on="order_id", how="inner")
    
    if len(df) == 0:
        print(f"Found {len(predictions_df)} predictions, but none have been delivered yet.")
        print("\nThis happens when:")
        print("1. Predictions have been made (✓)")
        print("2. But those orders haven't been delivered yet (wait for delivery simulation)")
        return
    
    print(f"Found {len(df)} completed orders with predictions\n")
    
    # Calculate error metrics
    df = df.with_columns([
        (pl.col("predicted_delivery_minutes") - pl.col("actual_delivery_minutes")).alias("error"),
        (pl.col("predicted_delivery_minutes") - pl.col("actual_delivery_minutes")).abs().alias("absolute_error"),
        ((pl.col("predicted_delivery_minutes") - pl.col("actual_delivery_minutes")).abs() / 
         pl.col("actual_delivery_minutes") * 100).alias("percentage_error"),
    ])
    
    # Overall metrics
    print("Overall Performance:")
    print(f"  Mean Absolute Error (MAE):  {df['absolute_error'].mean():.2f} minutes")
    print(f"  Root Mean Square Error:      {(df['error'].pow(2).mean() ** 0.5):.2f} minutes")
    print(f"  Mean Percentage Error:       {df['percentage_error'].mean():.2f}%")
    print(f"  Median Absolute Error:       {df['absolute_error'].median():.2f} minutes")
    print()
    
    # Accuracy bands
    within_5_min = (df['absolute_error'] <= 5).sum()
    within_10_min = (df['absolute_error'] <= 10).sum()
    within_15_min = (df['absolute_error'] <= 15).sum()
    total = len(df)
    
    print("Accuracy Bands:")
    print(f"  Within 5 minutes:  {within_5_min:4d} ({within_5_min/total*100:.1f}%)")
    print(f"  Within 10 minutes: {within_10_min:4d} ({within_10_min/total*100:.1f}%)")
    print(f"  Within 15 minutes: {within_15_min:4d} ({within_15_min/total*100:.1f}%)")
    print()
    
    # Best and worst predictions
    print("Best Predictions (lowest absolute error):")
    best = df.sort("absolute_error").head(5).select([
        "order_id",
        "predicted_delivery_minutes",
        "actual_delivery_minutes",
        "absolute_error"
    ])
    print(best)
    print()
    
    print("Worst Predictions (highest absolute error):")
    worst = df.sort("absolute_error", descending=True).head(5).select([
        "order_id",
        "predicted_delivery_minutes",
        "actual_delivery_minutes",
        "absolute_error"
    ])
    print(worst)
    print()
    
    # Performance by store
    store_performance = df.group_by("store_id").agg([
        pl.count().alias("num_orders"),
        pl.col("absolute_error").mean().alias("avg_error"),
        pl.col("predicted_delivery_minutes").mean().alias("avg_predicted"),
        pl.col("actual_delivery_minutes").mean().alias("avg_actual"),
    ]).sort("avg_error")
    
    print("Performance by Store (top 10 best, bottom 5 worst):")
    print("\nBest performing stores:")
    print(store_performance.head(10))
    print("\nWorst performing stores:")
    print(store_performance.tail(5))
    print()
    
    # Distribution statistics
    print("Error Distribution:")
    print(df.select(["error", "absolute_error"]).describe())
    print()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze prediction accuracy and export data"
    )
    parser.add_argument(
        "--predictions-db",
        type=Path,
        default=None,
        help="Path to predictions database (default: predictions.db)",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=None,
        help="Path to source database (default: from .env)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export predictions database tables to CSV files",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Output directory for CSV exports (default: exports/)",
    )
    
    args = parser.parse_args()
    
    if args.export:
        export_predictions_to_csv(args.predictions_db, args.export_dir)
    else:
        analyze_prediction_accuracy(args.predictions_db, args.source_db)
        
        print("\n" + "="*60)
        print("TIP: To improve accuracy, consider:")
        print("  1. Retraining the model with more recent data")
        print("  2. Adding more features (weather, traffic, etc.)")
        print("  3. Using more sophisticated models")
        print("  4. Feature engineering based on error patterns")
        print("="*60)


if __name__ == "__main__":
    main()
