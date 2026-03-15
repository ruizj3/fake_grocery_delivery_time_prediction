#!/usr/bin/env python3
"""Diagnose why the model has poor R² performance.

This script analyzes:
1. Data quality and distribution
2. Feature correlations with target
3. Feature importance from trained model
4. Potential data leakage or issues
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Database path from env
db_path = "/Users/josephruiz/Documents/GitHub/fake_grocery_delivery_service/database/grocery_delivery.db"

print("="*80)
print("MODEL PERFORMANCE DIAGNOSTIC")
print("="*80)

# 1. Check data distribution
print("\n1. DATA DISTRIBUTION ANALYSIS")
print("-"*80)

conn = sqlite3.connect(db_path)

# Basic stats
query = """
SELECT 
    COUNT(*) as total_orders,
    COUNT(CASE WHEN delivered_at IS NOT NULL THEN 1 END) as delivered_orders,
    AVG(CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL)) as avg_delivery_mins,
    MIN(CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL)) as min_delivery,
    MAX(CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL)) as max_delivery
FROM orders
WHERE delivered_at IS NOT NULL
"""

df_stats = pd.read_sql_query(query, conn)
print("\nOverall Statistics:")
print(df_stats.to_string(index=False))

# Check target variance - R² requires variance in the target (compute std in pandas)
query_variance = """
SELECT CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL) as delivery_mins
FROM orders
WHERE delivered_at IS NOT NULL
    AND created_at >= datetime('now', '-31 days')
LIMIT 10000
"""
df_variance_raw = pd.read_sql_query(query_variance, conn)
print(f"\nTarget Variance (sample of {len(df_variance_raw)} orders from last 31 days):")
print(f"  Mean: {df_variance_raw['delivery_mins'].mean():.2f} minutes")
print(f"  Std:  {df_variance_raw['delivery_mins'].std():.2f} minutes")
print(f"  Min:  {df_variance_raw['delivery_mins'].min():.2f} minutes")
print(f"  Max:  {df_variance_raw['delivery_mins'].max():.2f} minutes")
range_ratio = (df_variance_raw['delivery_mins'].max() - df_variance_raw['delivery_mins'].min()) / df_variance_raw['delivery_mins'].mean()
print(f"  Range/Mean Ratio: {range_ratio:.2f}")

range_ratio = (df_variance_raw['delivery_mins'].max() - df_variance_raw['delivery_mins'].min()) / df_variance_raw['delivery_mins'].mean()
print(f"  Range/Mean Ratio: {range_ratio:.2f}")

if df_variance_raw['delivery_mins'].std() == 0 or df_variance_raw['delivery_mins'].std() < 1:
    print("\n⚠️  WARNING: Target has ZERO or near-zero variance!")
    print("   This will result in poor R² scores.")
    print("   Possible causes:")
    print("   - All delivery times are the same (simulated data issue)")
    print("   - Timestamps are wrong")
    print("   - Data generation logic is broken")

# 2. Check feature distributions
print("\n\n2. FEATURE DISTRIBUTION ANALYSIS")
print("-"*80)

# Sample recent data
query_sample = """
SELECT 
    o.order_id,
    CAST((JULIANDAY(o.delivered_at) - JULIANDAY(o.created_at)) * 24 * 60 AS REAL) as delivery_time_minutes,
    o.total,
    COALESCE(oi.quantity, 1) as quantity,
    CAST(strftime('%H', o.created_at) AS INTEGER) as hour_of_day,
    CAST(strftime('%w', o.created_at) AS INTEGER) as day_of_week
FROM orders o
LEFT JOIN (SELECT order_id, SUM(quantity) AS quantity FROM order_items GROUP BY order_id) oi 
    ON o.order_id = oi.order_id
WHERE o.delivered_at IS NOT NULL
    AND o.created_at >= datetime('now', '-31 days')
LIMIT 5000
"""

df_sample = pd.read_sql_query(query_sample, conn)
print(f"\nSample size: {len(df_sample)} orders")
print("\nFeature Statistics:")
print(df_sample.describe())

# 3. Check correlations
print("\n\n3. FEATURE CORRELATIONS WITH TARGET")
print("-"*80)

if len(df_sample) > 0:
    # Drop non-numeric columns before correlation
    numeric_df = df_sample.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['delivery_time_minutes'].sort_values(ascending=False)
    print(corr)
    
    print("\n⚠️  CORRELATION ANALYSIS:")
    strong_corr = corr[abs(corr) > 0.3]
    if len(strong_corr) <= 1:  # Only target with itself
        print("   NO strong correlations found (|r| > 0.3)!")
        print("   This suggests features have little predictive power.")
        print("   Possible causes:")
        print("   - Features are not related to delivery time")
        print("   - Simulated data doesn't reflect real patterns")
        print("   - Missing important features (distance, traffic, etc.)")

# 4. Check if delivery time varies by features
print("\n\n4. DELIVERY TIME VARIANCE BY FEATURE")
print("-"*80)

# By hour of day
hour_query = """
SELECT 
    CAST(strftime('%H', created_at) AS INTEGER) as hour,
    COUNT(*) as count,
    AVG(CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL)) as avg_delivery,
    MIN(CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL)) as min_delivery,
    MAX(CAST((JULIANDAY(delivered_at) - JULIANDAY(created_at)) * 24 * 60 AS REAL)) as max_delivery
FROM orders
WHERE delivered_at IS NOT NULL
    AND created_at >= datetime('now', '-7 days')
GROUP BY hour
ORDER BY hour
"""
df_hour = pd.read_sql_query(hour_query, conn)
if len(df_hour) > 0:
    hour_variance = df_hour['avg_delivery'].std()
    print(f"\nDelivery time variance by hour: {hour_variance:.2f} minutes std")
    print(f"Range: {df_hour['avg_delivery'].min():.2f} - {df_hour['avg_delivery'].max():.2f} minutes")
    
    if hour_variance < 5:
        print("⚠️  Very low variance across hours! Delivery times don't vary much by time of day.")

# 5. Check training data issues
print("\n\n5. POTENTIAL DATA ISSUES")
print("-"*80)

# Check for constant values
issues = []

# Check if all delivery times are similar
if df_sample['delivery_time_minutes'].std() < 10:
    issues.append("All delivery times are very similar (std < 10 min)")

# Check if order totals vary
if df_sample['total'].std() < 100:
    issues.append("Order totals don't vary much")

# Check if quantity is mostly constant
if df_sample['quantity'].nunique() <= 3:
    issues.append(f"Quantity has only {df_sample['quantity'].nunique()} unique values")

# Check for missing aggregated features (which might be causing the issue)
print("\nChecking for restaurant aggregated features...")
rest_query = """
SELECT COUNT(*) as count FROM ml_restaurant_features
WHERE computed_at >= datetime('now', '-7 days')
"""
rest_features = pd.read_sql_query(rest_query, conn)
print(f"Recent restaurant features computed: {rest_features['count'].values[0]}")

if rest_features['count'].values[0] == 0:
    issues.append("NO restaurant aggregated features found - this is critical!")

if issues:
    print("\n⚠️  CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\n✅ No obvious data issues detected")

# 6. RECOMMENDATIONS
print("\n\n" + "="*80)
print("RECOMMENDATIONS FOR IMPROVING R² = 0.1")
print("="*80)

print("""
Current R² of 0.1 means the model explains only 10% of variance.

Based on the analysis above, likely issues with simulated data:

1. **Insufficient Feature Variance**
   - Simulated delivery times may not vary realistically
   - Add more realistic variation based on:
     * Distance (longer distance → longer delivery)
     * Time of day (rush hour → longer delivery)
     * Weather conditions
     * Store capacity/busyness

2. **Missing Key Features**
   - Distance is likely the most predictive feature
   - Historical restaurant performance is important
   - Current features may not capture delivery complexity

3. **Model Configuration**
   - With only R²=0.1, increase model complexity:
     * n_estimators: 100 → 300+
     * max_depth: 6 → 10
     * learning_rate: 0.1 → 0.05 (with more trees)

4. **Data Generation Issues**
   - Review how delivery times are simulated
   - Ensure realistic patterns:
     * Distance × speed + prep_time + wait_time
     * Add random noise for realism
     * Include rush hour multipliers

5. **Feature Engineering**
   - Create interaction features:
     * distance × hour_of_day
     * total × quantity (order complexity)
     * is_rush_hour (7-9am, 5-8pm)
   - Add temporal features:
     * days_since_store_opened
     * is_holiday

Next steps:
1. Review simulated data generation in your grocery delivery service
2. Add distance-based delivery time simulation
3. Retrain with enhanced hyperparameters
4. Consider using polynomial features or interactions
""")

conn.close()
print("\n" + "="*80)
