#!/usr/bin/env python3
"""Example: Automated drift monitoring with model retraining.

This script demonstrates how to:
1. Check for drift automatically
2. Generate alerts when anomalies detected
3. Trigger model retraining if needed
"""

from datetime import datetime, timedelta

from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor
from delivery_ml.training.pipeline import train_model


def main():
    print("=" * 80)
    print("AUTOMATED DRIFT MONITORING EXAMPLE")
    print("=" * 80)

    # Initialize drift monitor
    monitor = AutoDriftMonitor(
        reference_window_days=30,  # Compare against last 30 days
        current_window_days=7,  # Check recent 7 days
        check_interval_hours=24,  # Run daily
        alert_on_n_features=3,  # Alert if 3+ features drift
    )

    try:
        # Step 1: Run drift check
        print("\n[Step 1] Running drift check...")
        result = monitor.run_drift_check(force=True)

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return

        print(f"✓ Check complete - ID: {result['check_id']}")
        print(f"  Drift detected: {result['drift_detected']}")
        print(f"  Drifted tests: {result['drifted_count']}")

        # Step 2: Check for unacknowledged alerts
        print("\n[Step 2] Checking for alerts...")
        alerts = monitor.get_unacknowledged_alerts()

        if not alerts:
            print("✓ No alerts - model is healthy!")
            return

        print(f"⚠️  Found {len(alerts)} unacknowledged alerts:")
        for alert in alerts:
            severity_emoji = "🚨" if alert["severity"] == "critical" else "⚠️"
            print(f"  {severity_emoji} {alert['feature']}: {alert['message']}")

        # Step 3: Decide whether to retrain
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]

        if critical_alerts:
            print(
                f"\n[Step 3] 🚨 CRITICAL DRIFT DETECTED ({len(critical_alerts)} features)"
            )
            print("Triggering model retraining...")

            # Retrain on last 30 days
            train_start = datetime.now() - timedelta(days=30)
            train_end = datetime.now()

            print(f"Training period: {train_start.date()} to {train_end.date()}")

            # Note: Uncomment to actually retrain
            # train_model(train_start=train_start, train_end=train_end)
            print(
                "⚠️  Retraining skipped (uncomment train_model() to enable)"
            )

            # Acknowledge alerts after handling
            for alert in alerts:
                monitor.acknowledge_alert(alert["alert_id"])

            print("✓ Alerts acknowledged")
        else:
            print("\n[Step 3] Warning-level drift detected")
            print("No immediate action needed - monitoring...")

        # Step 4: Show monitoring summary
        print("\n[Step 4] Drift monitoring summary:")
        summary = monitor.get_drift_summary()

        print(f"  Total checks: {summary['total_checks']}")
        print(f"  Checks with drift: {summary['checks_with_drift']}")
        print(f"  Drift rate: {summary['drift_rate']:.1%}")
        print(f"  Unacknowledged alerts: {summary['unacknowledged_alerts']}")

    finally:
        monitor.close()

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run 'python check_drift.py --status' to see monitoring status")
    print("2. Run 'python check_drift.py --history' to see drift history")
    print("3. Run 'python scheduled_drift_monitor.py' for continuous monitoring")
    print("\nSee DRIFT_MONITORING.md for full documentation")


if __name__ == "__main__":
    main()
