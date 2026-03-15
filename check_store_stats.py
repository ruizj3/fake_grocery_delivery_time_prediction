#!/usr/bin/env python3
"""Check mean and std of delivery times per store."""

import sqlite3
from pathlib import Path
import polars as pl
from delivery_ml.config import settings


def analyze_delivery_times_by_store():
    """Analyze delivery time statistics per store."""
    
    print("=" * 80)
    print("DELIVERY TIME ANALYSIS BY STORE")
    print("=" * 80)
    print()
    
    # Connect to database
    db_path = settings.sqlite_db_path
    print(f"Database: {db_path}")
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Query orders with delivered_at timestamp
    query = """
    SELECT 
        o.store_id,
        o.order_id,
        o.created_at,
        o.confirmed_at,
        o.delivered_at,
        (julianday(o.delivered_at) - julianday(o.created_at)) * 24 * 60 as delivery_time_minutes
    FROM orders o
    WHERE o.delivered_at IS NOT NULL
        AND o.created_at IS NOT NULL
    ORDER BY o.store_id, o.created_at
    """
    
    print("Querying orders with delivered_at timestamp...")
    df = pl.read_database(query, connection=conn)
    
    print(f"Total delivered orders: {len(df)}")
    print()
    
    if df.is_empty():
        print("❌ No delivered orders found!")
        conn.close()
        return
    
    # Overall statistics
    print("=" * 80)
    print("OVERALL DELIVERY TIME STATISTICS")
    print("=" * 80)
    
    overall_stats = df.select("delivery_time_minutes").describe()
    print(overall_stats)
    print()
    
    # Per-store statistics
    print("=" * 80)
    print("PER-STORE DELIVERY TIME STATISTICS")
    print("=" * 80)
    print()
    
    store_stats = df.group_by("store_id").agg([
        pl.count("order_id").alias("order_count"),
        pl.mean("delivery_time_minutes").alias("mean_delivery_minutes"),
        pl.std("delivery_time_minutes").alias("std_delivery_minutes"),
        pl.min("delivery_time_minutes").alias("min_delivery_minutes"),
        pl.max("delivery_time_minutes").alias("max_delivery_minutes"),
    ]).sort("order_count", descending=True)
    
    print(f"{'Store ID':<15} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)
    
    for row in store_stats.iter_rows(named=True):
        print(f"{row['store_id']:<15} {row['order_count']:>8} "
              f"{row['mean_delivery_minutes']:>10.2f} {row['std_delivery_minutes']:>10.2f} "
              f"{row['min_delivery_minutes']:>10.2f} {row['max_delivery_minutes']:>10.2f}")
    
    print()
    
    # Check for issues
    print("=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)
    print()
    
    # Check for stores with zero variation
    zero_std_stores = store_stats.filter(pl.col("std_delivery_minutes") == 0)
    if len(zero_std_stores) > 0:
        print(f"⚠️  WARNING: {len(zero_std_stores)} stores have ZERO variation in delivery times:")
        for row in zero_std_stores.iter_rows(named=True):
            print(f"  Store {row['store_id']}: {row['order_count']} orders, all = {row['mean_delivery_minutes']:.2f} min")
        print()
    
    # Check for stores with very low variation
    low_std_stores = store_stats.filter(
        (pl.col("std_delivery_minutes") > 0) & (pl.col("std_delivery_minutes") < 1)
    )
    if len(low_std_stores) > 0:
        print(f"⚠️  WARNING: {len(low_std_stores)} stores have very LOW variation (std < 1 minute):")
        for row in low_std_stores.iter_rows(named=True):
            print(f"  Store {row['store_id']}: std = {row['std_delivery_minutes']:.3f} min")
        print()
    
    # Check for stores with only 1 order
    single_order_stores = store_stats.filter(pl.col("order_count") == 1)
    if len(single_order_stores) > 0:
        print(f"⚠️  WARNING: {len(single_order_stores)} stores have only 1 order")
        print()
    
    # Check overall variation
    overall_std = df["delivery_time_minutes"].std()
    overall_mean = df["delivery_time_minutes"].mean()
    
    print(f"Overall Mean: {overall_mean:.2f} minutes")
    print(f"Overall Std:  {overall_std:.2f} minutes")
    print(f"Coefficient of Variation: {(overall_std/overall_mean)*100:.1f}%")
    print()
    
    if overall_std < 1:
        print("❌ CRITICAL: Overall standard deviation is < 1 minute!")
        print("   This means delivery times barely vary - model cannot learn!")
    elif overall_std < 5:
        print("⚠️  WARNING: Overall standard deviation is very low (< 5 minutes)")
        print("   Limited variation makes learning difficult")
    else:
        print("✓ Sufficient variation in delivery times")
    
    print()
    
    # Check for duplicate delivery times (potential data generation issue)
    print("=" * 80)
    print("DUPLICATE VALUE ANALYSIS")
    print("=" * 80)
    print()
    
    value_counts = df.group_by("delivery_time_minutes").agg(
        pl.count().alias("count")
    ).sort("count", descending=True)
    
    print(f"Total unique delivery times: {len(value_counts)}")
    print(f"Most common delivery times:")
    print(value_counts.head(10))
    print()
    
    # If too many duplicates, it's a problem
    max_duplicates = value_counts["count"].max()
    if max_duplicates > len(df) * 0.1:
        print(f"⚠️  WARNING: Most common value appears {max_duplicates} times ({max_duplicates/len(df)*100:.1f}%)")
        print("   This suggests limited variation in the data")
    
    conn.close()
    
    # Save report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"store_delivery_analysis_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DELIVERY TIME ANALYSIS BY STORE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Database: {db_path}\n")
        f.write(f"Total delivered orders: {len(df)}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"Mean: {overall_mean:.2f} minutes\n")
        f.write(f"Std:  {overall_std:.2f} minutes\n")
        f.write(f"CV:   {(overall_std/overall_mean)*100:.1f}%\n\n")
        
        f.write("PER-STORE STATISTICS:\n")
        f.write(f"{'Store ID':<15} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
        f.write("-" * 80 + "\n")
        
        for row in store_stats.iter_rows(named=True):
            f.write(f"{row['store_id']:<15} {row['order_count']:>8} "
                   f"{row['mean_delivery_minutes']:>10.2f} {row['std_delivery_minutes']:>10.2f} "
                   f"{row['min_delivery_minutes']:>10.2f} {row['max_delivery_minutes']:>10.2f}\n")
    
    print(f"✅ Analysis saved to: {output_file}")


if __name__ == "__main__":
    analyze_delivery_times_by_store()
