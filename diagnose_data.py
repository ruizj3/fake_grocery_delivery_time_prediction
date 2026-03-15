#!/usr/bin/env python3
"""Diagnose data quality and feature issues for delivery time prediction."""

from datetime import datetime, timedelta
import polars as pl
from delivery_ml.features.store import FeatureStore
from delivery_ml.features.definitions import TRAINING_FEATURES


def diagnose_training_data():
    """Comprehensive data quality diagnostics."""
    
    print("=" * 80)
    print("DATA QUALITY DIAGNOSTICS")
    print("=" * 80)
    print()
    
    # Initialize feature store
    store = FeatureStore()
    store.initialize()
    
    # Get date range
    min_date, max_date = store.offline.get_date_range()
    print(f"Data Range: {min_date} to {max_date}")
    print(f"Total Days: {(max_date - min_date).days}")
    print()
    
    # Get training data (last 31 days)
    train_end = max_date
    train_start = max_date - timedelta(days=31)
    
    print(f"Fetching training data: {train_start} to {train_end}")
    df = store.offline.get_training_data(train_start, train_end)
    
    print(f"Total Samples: {len(df)}")
    print()
    
    if df.is_empty():
        print("❌ ERROR: No training data found!")
        return
    
    # 1. Check target variable
    print("=" * 80)
    print("TARGET VARIABLE ANALYSIS (delivery_time_minutes)")
    print("=" * 80)
    
    target_stats = df.select("delivery_time_minutes").describe()
    print(target_stats)
    print()
    
    # Check for issues
    target_values = df["delivery_time_minutes"].to_list()
    unique_values = set(target_values)
    
    print(f"Unique Values: {len(unique_values)}")
    print(f"Min: {min(target_values):.2f}")
    print(f"Max: {max(target_values):.2f}")
    print(f"Mean: {sum(target_values)/len(target_values):.2f}")
    print()
    
    if len(unique_values) == 1:
        print("⚠️  WARNING: Target has only ONE unique value - no variation to learn!")
    elif len(unique_values) < 10:
        print(f"⚠️  WARNING: Target has very few unique values ({len(unique_values)})")
        print(f"Values: {sorted(unique_values)}")
    
    print()
    
    # 2. Check features
    print("=" * 80)
    print("FEATURE ANALYSIS")
    print("=" * 80)
    print()
    
    for feature in TRAINING_FEATURES:
        if feature in df.columns:
            values = df[feature].to_list()
            unique = set(values)
            
            # Count nulls/missing
            null_count = sum(1 for v in values if v is None or (isinstance(v, float) and v != v))
            
            print(f"{feature}:")
            print(f"  Unique values: {len(unique)}")
            print(f"  Nulls: {null_count}")
            
            if len(unique) == 1:
                print(f"  ⚠️  CONSTANT - Only value: {list(unique)[0]}")
            elif len(unique) < 5 and len(unique) > 0:
                print(f"  Values: {sorted(unique)[:10]}")
            else:
                try:
                    numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
                    if numeric_values:
                        print(f"  Min: {min(numeric_values):.2f}, Max: {max(numeric_values):.2f}, Mean: {sum(numeric_values)/len(numeric_values):.2f}")
                except:
                    pass
            print()
        else:
            print(f"{feature}: ❌ MISSING FROM DATA")
            print()
    
    # 3. Check for correlations
    print("=" * 80)
    print("FEATURE-TARGET CORRELATIONS")
    print("=" * 80)
    print()
    
    # Convert to pandas for correlation
    pdf = df.select([*TRAINING_FEATURES, "delivery_time_minutes"]).to_pandas()
    
    # Replace inf/-inf with NaN, then fill NaN with -1
    pdf = pdf.replace([float('inf'), float('-inf')], float('nan')).fillna(-1)
    
    correlations = pdf.corr()["delivery_time_minutes"].drop("delivery_time_minutes").sort_values(ascending=False)
    
    print("Top positive correlations:")
    print(correlations.head(5))
    print()
    print("Top negative correlations:")
    print(correlations.tail(5))
    print()
    
    if abs(correlations).max() < 0.1:
        print("⚠️  WARNING: No feature has strong correlation with target (all < 0.1)")
        print("   This explains why models can't learn anything!")
    
    # 4. Check data distribution
    print("=" * 80)
    print("DATA DISTRIBUTION ISSUES")
    print("=" * 80)
    print()
    
    # Check for train/test split issues
    test_size = int(len(df) * 0.2)
    train_size = len(df) - test_size
    
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")
    print()
    
    if test_size < 50:
        print("⚠️  WARNING: Test set is very small - metrics may be unreliable")
    
    # 5. Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    issues = []
    
    if len(unique_values) < 10:
        issues.append("• Target variable has low variation - check data generation")
    
    if abs(correlations).max() < 0.1:
        issues.append("• Features have no correlation with target - need better features")
    
    if len(df) < 100:
        issues.append("• Insufficient training data - need more samples")
    
    # Check for constant features
    constant_features = []
    for feature in TRAINING_FEATURES:
        if feature in df.columns:
            values = df[feature].to_list()
            if len(set(values)) == 1:
                constant_features.append(feature)
    
    if constant_features:
        issues.append(f"• Constant features detected: {', '.join(constant_features)}")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("✓ No obvious data quality issues found")
        print("  Problem may be:")
        print("  • Point-in-time correctness issues")
        print("  • Feature engineering needed")
        print("  • Target transformation needed")
    
    print()
    
    # Save diagnostic report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data_diagnostics_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Data Range: {min_date} to {max_date}\n")
        f.write(f"Total Samples: {len(df)}\n\n")
        f.write("TARGET STATISTICS:\n")
        f.write(str(target_stats) + "\n\n")
        f.write("FEATURE CORRELATIONS:\n")
        f.write(str(correlations) + "\n\n")
        if issues:
            f.write("ISSUES FOUND:\n")
            for issue in issues:
                f.write(issue + "\n")
    
    print(f"✅ Diagnostic report saved to: {output_file}")


if __name__ == "__main__":
    diagnose_training_data()
