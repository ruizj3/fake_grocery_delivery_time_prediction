#!/usr/bin/env python3
"""Quick test to verify drift monitoring is working.

This test creates synthetic data with drift and verifies
the drift detection system catches it.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from delivery_ml.monitoring.drift import DriftDetector


def test_basic_drift_detection():
    """Test that drift detector catches obvious drift."""
    print("Testing basic drift detection...")

    # Create reference data (normal distribution)
    reference_df = pl.DataFrame(
        {
            "haversine_distance": [5.0 + i * 0.1 for i in range(100)],
            "total": [2000.0 + i * 10 for i in range(100)],
            "quantity": [3.0] * 100,
        }
    )

    # Create drifted data (shifted distribution)
    current_df = pl.DataFrame(
        {
            "haversine_distance": [
                8.0 + i * 0.1 for i in range(100)
            ],  # Shifted by 3
            "total": [2500.0 + i * 10 for i in range(100)],  # Shifted by 500
            "quantity": [3.0] * 100,  # No drift
        }
    )

    # Run drift detection
    detector = DriftDetector(p_value_threshold=0.05)

    # Test haversine_distance
    ref_values = reference_df["haversine_distance"].to_list()
    cur_values = current_df["haversine_distance"].to_list()

    result = detector.detect_drift_ks_test(ref_values, cur_values, "haversine_distance")

    print(f"\nHaversine Distance Drift Test:")
    print(f"  Reference mean: {result.reference_mean:.2f}")
    print(f"  Current mean: {result.current_mean:.2f}")
    print(f"  P-value: {result.p_value:.6f}")
    print(f"  Drift detected: {result.is_drifted}")

    assert result.is_drifted, "Should detect drift in haversine_distance"

    # Test total
    ref_values = reference_df["total"].to_list()
    cur_values = current_df["total"].to_list()

    result = detector.detect_drift_ks_test(ref_values, cur_values, "total")

    print(f"\nTotal Drift Test:")
    print(f"  Reference mean: {result.reference_mean:.2f}")
    print(f"  Current mean: {result.current_mean:.2f}")
    print(f"  P-value: {result.p_value:.6f}")
    print(f"  Drift detected: {result.is_drifted}")

    assert result.is_drifted, "Should detect drift in total"

    # Test quantity (no drift)
    ref_values = reference_df["quantity"].to_list()
    cur_values = current_df["quantity"].to_list()

    result = detector.detect_drift_ks_test(ref_values, cur_values, "quantity")

    print(f"\nQuantity Drift Test:")
    print(f"  Reference mean: {result.reference_mean:.2f}")
    print(f"  Current mean: {result.current_mean:.2f}")
    print(f"  P-value: {result.p_value:.6f}")
    print(f"  Drift detected: {result.is_drifted}")

    assert not result.is_drifted, "Should NOT detect drift in quantity"

    print("\n✅ Basic drift detection test passed!")


def test_psi_drift_detection():
    """Test PSI (Population Stability Index) drift detection."""
    print("\nTesting PSI drift detection...")

    # Create reference data
    reference_values = [5.0 + i * 0.1 for i in range(100)]

    # Create drifted data
    current_values = [7.0 + i * 0.1 for i in range(100)]

    detector = DriftDetector()
    result = detector.detect_drift_psi(
        reference_values, current_values, "test_feature", psi_threshold=0.2
    )

    print(f"PSI Test:")
    print(f"  PSI statistic: {result.statistic:.4f}")
    print(f"  Threshold: {result.threshold}")
    print(f"  Drift detected: {result.is_drifted}")

    assert result.is_drifted, "Should detect drift via PSI"

    print("✅ PSI drift detection test passed!")


def test_drift_report_generation():
    """Test drift report generation."""
    print("\nTesting drift report generation...")

    reference_df = pl.DataFrame(
        {
            "haversine_distance": [5.0 + i * 0.1 for i in range(100)],
            "total": [2000.0 + i * 10 for i in range(100)],
        }
    )

    current_df = pl.DataFrame(
        {
            "haversine_distance": [8.0 + i * 0.1 for i in range(100)],
            "total": [2500.0 + i * 10 for i in range(100)],
        }
    )

    detector = DriftDetector()
    results = detector.check_all_features(reference_df, current_df)
    report = detector.generate_report(results)

    print(f"Drift Report:")
    print(f"  Total tests: {report['total_tests']}")
    print(f"  Drifted count: {report['drifted_count']}")
    print(f"  Drift detected: {report['drift_detected']}")

    assert report["drift_detected"], "Should detect drift"
    assert report["drifted_count"] > 0, "Should have drifted features"

    print(f"\nDrifted features:")
    for feature in report["drifted_features"]:
        print(f"  - {feature['feature']} ({feature['test']})")

    print("✅ Drift report generation test passed!")


def main():
    print("=" * 80)
    print("DRIFT MONITORING SYSTEM TEST")
    print("=" * 80)

    try:
        test_basic_drift_detection()
        test_psi_drift_detection()
        test_drift_report_generation()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nDrift monitoring system is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python check_drift.py' to check for real drift")
        print("2. Run 'python examples/drift_monitoring_example.py' for full example")
        print("3. Run 'python scheduled_drift_monitor.py' for continuous monitoring")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
