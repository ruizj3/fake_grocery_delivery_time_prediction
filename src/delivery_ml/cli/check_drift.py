#!/usr/bin/env python3
"""CLI tool for manual drift checking and monitoring.

Usage:
    python check_drift.py                    # Run drift check now
    python check_drift.py --status           # Show drift monitoring status
    python check_drift.py --history          # Show drift history
    python check_drift.py --alerts           # Show unacknowledged alerts
    python check_drift.py --ack ALERT_ID     # Acknowledge alert
"""

import argparse
import sys

from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor


def main():
    parser = argparse.ArgumentParser(description="Drift monitoring CLI")
    parser.add_argument("--force", action="store_true", help="Force drift check now")
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    parser.add_argument("--history", action="store_true", help="Show drift history")
    parser.add_argument(
        "--alerts", action="store_true", help="Show unacknowledged alerts"
    )
    parser.add_argument("--ack", type=int, metavar="ALERT_ID", help="Acknowledge alert")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days for history (default: 30)"
    )

    args = parser.parse_args()

    monitor = AutoDriftMonitor()

    try:
        if args.status:
            show_status(monitor)
        elif args.history:
            show_history(monitor, args.days)
        elif args.alerts:
            show_alerts(monitor)
        elif args.ack:
            acknowledge_alert(monitor, args.ack)
        else:
            run_check(monitor, args.force)
    finally:
        monitor.close()


def show_status(monitor: AutoDriftMonitor):
    """Show drift monitoring status."""
    print("\n" + "=" * 80)
    print("DRIFT MONITORING STATUS")
    print("=" * 80)

    summary = monitor.get_drift_summary()

    print(f"Total Checks Run: {summary['total_checks']}")
    print(f"Checks with Drift: {summary['checks_with_drift']}")
    print(f"Drift Rate: {summary['drift_rate']:.1%}")
    print(f"\nLatest Check: {summary['latest_check']}")
    print(f"Drift Detected: {summary['latest_drift_detected']}")
    print(f"Drifted Features: {summary['latest_drifted_features']}")
    print(f"\nUnacknowledged Alerts: {summary['unacknowledged_alerts']}")
    print(f"Next Check Due: {summary['next_check_due']}")


def show_history(monitor: AutoDriftMonitor, days: int):
    """Show drift check history."""
    print("\n" + "=" * 80)
    print(f"DRIFT HISTORY (Last {days} days)")
    print("=" * 80)

    history = monitor.get_drift_history(days=days)

    if len(history) == 0:
        print("No drift checks found.")
        return

    print(f"\nFound {len(history)} checks:\n")
    print(
        f"{'Check ID':<10} {'Timestamp':<20} {'Drift?':<10} {'Drifted Features':<20}"
    )
    print("-" * 80)

    for row in history.iter_rows(named=True):
        drift_status = "✓ YES" if row["drift_detected"] else "✗ No"
        print(
            f"{row['check_id']:<10} {row['check_timestamp'][:19]:<20} "
            f"{drift_status:<10} {row['drifted_features_count']:<20}"
        )


def show_alerts(monitor: AutoDriftMonitor):
    """Show unacknowledged alerts."""
    print("\n" + "=" * 80)
    print("UNACKNOWLEDGED DRIFT ALERTS")
    print("=" * 80)

    alerts = monitor.get_unacknowledged_alerts()

    if not alerts:
        print("\n✓ No unacknowledged alerts!")
        return

    print(f"\nFound {len(alerts)} unacknowledged alerts:\n")

    for alert in alerts:
        severity_emoji = "🚨" if alert["severity"] == "critical" else "⚠️"
        print(f"\nAlert ID: {alert['alert_id']} {severity_emoji}")
        print(f"Timestamp: {alert['timestamp']}")
        print(f"Severity: {alert['severity'].upper()}")
        print(f"Feature: {alert['feature']}")
        print(f"Test: {alert['test']}")
        if alert["drift_percentage"]:
            print(f"Drift: {alert['drift_percentage']:+.1f}%")
        print(f"Message: {alert['message']}")
        print("-" * 80)

    print(f"\nTo acknowledge an alert: python check_drift.py --ack ALERT_ID")


def acknowledge_alert(monitor: AutoDriftMonitor, alert_id: int):
    """Acknowledge an alert."""
    monitor.acknowledge_alert(alert_id)
    print(f"✓ Alert {alert_id} acknowledged")


def run_check(monitor: AutoDriftMonitor, force: bool):
    """Run drift check."""
    print("\n" + "=" * 80)
    print("RUNNING DRIFT CHECK")
    print("=" * 80)

    result = monitor.run_drift_check(force=force)

    if result.get("skipped"):
        print(f"\n⏭️  Check skipped: {result.get('reason')}")
        return

    if result.get("error"):
        print(f"\n❌ Error: {result.get('error')}")
        return

    print(f"\n✓ Check complete - Check ID: {result.get('check_id')}")
    print(f"Drift Detected: {result.get('drift_detected')}")
    print(f"Drifted Tests: {result.get('drifted_count')}")
    print(f"Alerts Generated: {result.get('alerts_generated')}")

    if result.get("drift_detected"):
        print("\n🚨 DRIFT DETECTED!")
        print("\nDrifted Features:")
        for feature in result["report"]["drifted_features"]:
            print(f"\n  Feature: {feature['feature']}")
            print(f"  Test: {feature['test']}")
            print(f"  Statistic: {feature['statistic']:.4f}")
            print(f"  P-value: {feature['p_value']:.4f}")
            if feature["reference_mean"] and feature["current_mean"]:
                pct = (
                    (feature["current_mean"] - feature["reference_mean"])
                    / feature["reference_mean"]
                    * 100
                )
                print(
                    f"  Mean: {feature['reference_mean']:.2f} → {feature['current_mean']:.2f} ({pct:+.1f}%)"
                )

        print("\nRecommendation: Consider retraining your model!")
    else:
        print("\n✓ No significant drift detected")


if __name__ == "__main__":
    main()
