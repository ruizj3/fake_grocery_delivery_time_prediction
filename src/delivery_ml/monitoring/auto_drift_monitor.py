"""Automated drift detection and anomaly reporting.

This module provides automated drift monitoring with:
- Scheduled drift checks
- Anomaly detection and reporting
- Email/Slack/logging alerts
- Drift history tracking
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from delivery_ml.config import settings
from delivery_ml.features.store import FeatureStore
from delivery_ml.monitoring.drift import DriftDetector, DriftResult

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift."""

    timestamp: datetime
    severity: str  # "warning", "critical"
    feature_name: str
    test_name: str
    statistic: float
    p_value: float
    reference_mean: float | None
    current_mean: float | None
    drift_percentage: float | None = None
    message: str = ""


class AutoDriftMonitor:
    """Automated drift monitoring with alerting."""

    def __init__(
        self,
        db_path: Path | None = None,
        reference_window_days: int = 30,
        current_window_days: int = 7,
        check_interval_hours: int = 24,
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        alert_on_n_features: int = 3,  # Alert if N or more features drift
    ):
        self.db_path = db_path or Path("drift_monitoring.db")
        self.reference_window_days = reference_window_days
        self.current_window_days = current_window_days
        self.check_interval_hours = check_interval_hours
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
        self.alert_on_n_features = alert_on_n_features

        self.feature_store = FeatureStore()
        self.drift_detector = DriftDetector(p_value_threshold=p_value_threshold)

        # Initialize database
        self._init_db()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _init_db(self) -> None:
        """Initialize drift monitoring database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Drift check history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_checks (
                check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_timestamp TEXT NOT NULL,
                reference_start TEXT NOT NULL,
                reference_end TEXT NOT NULL,
                current_start TEXT NOT NULL,
                current_end TEXT NOT NULL,
                total_features_checked INTEGER,
                drifted_features_count INTEGER,
                drift_detected BOOLEAN,
                report_json TEXT
            )
        """)

        # Individual feature drift results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id INTEGER,
                feature_name TEXT NOT NULL,
                test_name TEXT NOT NULL,
                statistic REAL,
                p_value REAL,
                is_drifted BOOLEAN,
                threshold REAL,
                reference_mean REAL,
                current_mean REAL,
                FOREIGN KEY (check_id) REFERENCES drift_checks (check_id)
            )
        """)

        # Drift alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id INTEGER,
                alert_timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                test_name TEXT NOT NULL,
                statistic REAL,
                p_value REAL,
                reference_mean REAL,
                current_mean REAL,
                drift_percentage REAL,
                message TEXT,
                acknowledged BOOLEAN DEFAULT 0,
                FOREIGN KEY (check_id) REFERENCES drift_checks (check_id)
            )
        """)

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_checks_timestamp ON drift_checks(check_timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_results_check ON drift_results(check_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_drift_alerts_check ON drift_alerts(check_id)"
        )

        conn.commit()
        conn.close()

    def should_run_check(self) -> bool:
        """Determine if enough time has passed since last check."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT check_timestamp 
            FROM drift_checks 
            ORDER BY check_timestamp DESC 
            LIMIT 1
        """)

        result = cursor.fetchone()
        conn.close()

        if not result:
            return True

        last_check = datetime.fromisoformat(result[0])
        hours_since_last = (datetime.now() - last_check).total_seconds() / 3600

        return hours_since_last >= self.check_interval_hours

    def run_drift_check(self, force: bool = False) -> dict[str, Any]:
        """Run automated drift check and generate alerts if needed."""
        if not force and not self.should_run_check():
            logger.info("Skipping drift check - not enough time since last check")
            return {"skipped": True, "reason": "Too soon since last check"}

        logger.info("Starting automated drift check...")

        # Define time windows
        now = datetime.now()
        reference_end = now - timedelta(days=self.current_window_days)
        reference_start = reference_end - timedelta(days=self.reference_window_days)
        current_start = now - timedelta(days=self.current_window_days)
        current_end = now

        # Get data
        logger.info(
            f"Reference period: {reference_start.date()} to {reference_end.date()}"
        )
        logger.info(f"Current period: {current_start.date()} to {current_end.date()}")

        reference_df = self.feature_store.offline.get_training_data(
            reference_start, reference_end
        )
        current_df = self.feature_store.offline.get_training_data(
            current_start, current_end
        )

        if reference_df is None or len(reference_df) < 100:
            logger.warning("Insufficient reference data for drift check")
            return {"error": "Insufficient reference data"}

        if current_df is None or len(current_df) < 100:
            logger.warning("Insufficient current data for drift check")
            return {"error": "Insufficient current data"}

        # Run drift detection
        logger.info(
            f"Checking drift on {len(reference_df)} reference and {len(current_df)} current samples"
        )
        results = self.drift_detector.check_all_features(reference_df, current_df)
        report = self.drift_detector.generate_report(results)

        # Save to database
        check_id = self._save_drift_check(
            reference_start,
            reference_end,
            current_start,
            current_end,
            results,
            report,
        )

        # Generate and save alerts
        alerts = self._generate_alerts(check_id, results, report)

        # Log summary
        logger.info(f"Drift check complete - Check ID: {check_id}")
        logger.info(
            f"Features checked: {report['total_tests']//2}"
        )  # Divide by 2 (KS + PSI)
        logger.info(f"Drift detected in: {report['drifted_count']} tests")
        logger.info(f"Alerts generated: {len(alerts)}")

        if report["drift_detected"]:
            logger.warning(f"⚠️  DRIFT DETECTED! Generated {len(alerts)} alerts")
            self._send_notifications(alerts, report)

        return {
            "check_id": check_id,
            "drift_detected": report["drift_detected"],
            "drifted_count": report["drifted_count"],
            "alerts_generated": len(alerts),
            "report": report,
        }

    def _save_drift_check(
        self,
        reference_start: datetime,
        reference_end: datetime,
        current_start: datetime,
        current_end: datetime,
        results: list[DriftResult],
        report: dict[str, Any],
    ) -> int:
        """Save drift check to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save check summary
        cursor.execute(
            """
            INSERT INTO drift_checks (
                check_timestamp, reference_start, reference_end,
                current_start, current_end, total_features_checked,
                drifted_features_count, drift_detected, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                reference_start.isoformat(),
                reference_end.isoformat(),
                current_start.isoformat(),
                current_end.isoformat(),
                len(results),
                report["drifted_count"],
                report["drift_detected"],
                json.dumps(report),
            ),
        )

        check_id = cursor.lastrowid

        # Save individual results
        for result in results:
            cursor.execute(
                """
                INSERT INTO drift_results (
                    check_id, feature_name, test_name, statistic,
                    p_value, is_drifted, threshold, reference_mean, current_mean
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    check_id,
                    result.feature_name,
                    result.test_name,
                    result.statistic,
                    result.p_value,
                    result.is_drifted,
                    result.threshold,
                    result.reference_mean,
                    result.current_mean,
                ),
            )

        conn.commit()
        conn.close()

        return check_id

    def _generate_alerts(
        self, check_id: int, results: list[DriftResult], report: dict[str, Any]
    ) -> list[DriftAlert]:
        """Generate alerts based on drift results."""
        alerts = []

        # Count unique features that drifted
        drifted_features = set()
        for r in results:
            if r.is_drifted:
                drifted_features.add(r.feature_name)

        # Determine overall severity
        if len(drifted_features) >= self.alert_on_n_features:
            overall_severity = "critical"
        elif len(drifted_features) > 0:
            overall_severity = "warning"
        else:
            return []

        # Create alerts for each drifted feature
        for result in results:
            if not result.is_drifted:
                continue

            # Calculate drift percentage if we have means
            drift_pct = None
            if result.reference_mean and result.current_mean:
                drift_pct = (
                    (result.current_mean - result.reference_mean)
                    / result.reference_mean
                    * 100
                )

            message = self._create_alert_message(result, drift_pct)

            alert = DriftAlert(
                timestamp=datetime.now(),
                severity=overall_severity,
                feature_name=result.feature_name,
                test_name=result.test_name,
                statistic=result.statistic,
                p_value=result.p_value,
                reference_mean=result.reference_mean,
                current_mean=result.current_mean,
                drift_percentage=drift_pct,
                message=message,
            )

            alerts.append(alert)

            # Save alert to database
            self._save_alert(check_id, alert)

        return alerts

    def _create_alert_message(
        self, result: DriftResult, drift_pct: float | None
    ) -> str:
        """Create human-readable alert message."""
        msg = f"Feature '{result.feature_name}' has drifted ({result.test_name})."

        if result.reference_mean and result.current_mean:
            msg += f" Mean changed from {result.reference_mean:.2f} to {result.current_mean:.2f}"

        if drift_pct:
            msg += f" ({drift_pct:+.1f}%)"

        msg += f". P-value: {result.p_value:.4f}"

        return msg

    def _save_alert(self, check_id: int, alert: DriftAlert) -> None:
        """Save alert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO drift_alerts (
                check_id, alert_timestamp, severity, feature_name,
                test_name, statistic, p_value, reference_mean,
                current_mean, drift_percentage, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                check_id,
                alert.timestamp.isoformat(),
                alert.severity,
                alert.feature_name,
                alert.test_name,
                alert.statistic,
                alert.p_value,
                alert.reference_mean,
                alert.current_mean,
                alert.drift_percentage,
                alert.message,
            ),
        )

        conn.commit()
        conn.close()

    def _send_notifications(
        self, alerts: list[DriftAlert], report: dict[str, Any]
    ) -> None:
        """Send notifications for drift alerts."""
        # Log alerts
        logger.warning("=" * 80)
        logger.warning("DRIFT ALERT SUMMARY")
        logger.warning("=" * 80)

        critical_alerts = [a for a in alerts if a.severity == "critical"]
        warning_alerts = [a for a in alerts if a.severity == "warning"]

        if critical_alerts:
            logger.error(f"🚨 CRITICAL: {len(critical_alerts)} features drifted")
        if warning_alerts:
            logger.warning(f"⚠️  WARNING: {len(warning_alerts)} features drifted")

        # Log each alert
        for alert in alerts:
            emoji = "🚨" if alert.severity == "critical" else "⚠️"
            logger.warning(f"{emoji} {alert.message}")

        logger.warning("=" * 80)

        # TODO: Add email notification
        # self._send_email_alert(alerts, report)

        # TODO: Add Slack notification
        # self._send_slack_alert(alerts, report)

    def get_drift_history(self, days: int = 30) -> pl.DataFrame:
        """Get drift check history."""
        conn = sqlite3.connect(self.db_path)

        cutoff = datetime.now() - timedelta(days=days)

        df = pl.read_database(
            f"""
            SELECT 
                check_id,
                check_timestamp,
                reference_start,
                reference_end,
                current_start,
                current_end,
                total_features_checked,
                drifted_features_count,
                drift_detected
            FROM drift_checks
            WHERE check_timestamp >= '{cutoff.isoformat()}'
            ORDER BY check_timestamp DESC
            """,
            connection=conn,
        )

        conn.close()
        return df

    def get_unacknowledged_alerts(self) -> list[dict[str, Any]]:
        """Get all unacknowledged drift alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                alert_id, check_id, alert_timestamp, severity,
                feature_name, test_name, message, drift_percentage
            FROM drift_alerts
            WHERE acknowledged = 0
            ORDER BY alert_timestamp DESC
        """)

        alerts = []
        for row in cursor.fetchall():
            alerts.append(
                {
                    "alert_id": row[0],
                    "check_id": row[1],
                    "timestamp": row[2],
                    "severity": row[3],
                    "feature": row[4],
                    "test": row[5],
                    "message": row[6],
                    "drift_percentage": row[7],
                }
            )

        conn.close()
        return alerts

    def acknowledge_alert(self, alert_id: int) -> None:
        """Mark an alert as acknowledged."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE drift_alerts 
            SET acknowledged = 1 
            WHERE alert_id = ?
        """,
            (alert_id,),
        )

        conn.commit()
        conn.close()

        logger.info(f"Alert {alert_id} acknowledged")

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of drift monitoring status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Latest check
        cursor.execute("""
            SELECT check_timestamp, drift_detected, drifted_features_count
            FROM drift_checks
            ORDER BY check_timestamp DESC
            LIMIT 1
        """)
        latest = cursor.fetchone()

        # Unacknowledged alerts
        cursor.execute("""
            SELECT COUNT(*) 
            FROM drift_alerts 
            WHERE acknowledged = 0
        """)
        unack_count = cursor.fetchone()[0]

        # Total checks
        cursor.execute("SELECT COUNT(*) FROM drift_checks")
        total_checks = cursor.fetchone()[0]

        # Checks with drift
        cursor.execute("SELECT COUNT(*) FROM drift_checks WHERE drift_detected = 1")
        drift_checks = cursor.fetchone()[0]

        conn.close()

        return {
            "total_checks": total_checks,
            "checks_with_drift": drift_checks,
            "drift_rate": drift_checks / total_checks if total_checks > 0 else 0,
            "latest_check": latest[0] if latest else None,
            "latest_drift_detected": bool(latest[1]) if latest else None,
            "latest_drifted_features": latest[2] if latest else None,
            "unacknowledged_alerts": unack_count,
            "next_check_due": self._next_check_time(),
        }

    def _next_check_time(self) -> str | None:
        """Calculate when next check is due."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT check_timestamp 
            FROM drift_checks 
            ORDER BY check_timestamp DESC 
            LIMIT 1
        """)

        result = cursor.fetchone()
        conn.close()

        if not result:
            return "Now"

        last_check = datetime.fromisoformat(result[0])
        next_check = last_check + timedelta(hours=self.check_interval_hours)

        if next_check <= datetime.now():
            return "Now"

        return next_check.isoformat()

    def close(self) -> None:
        """Clean up resources."""
        self.feature_store.close()


if __name__ == "__main__":
    # Run manual drift check
    monitor = AutoDriftMonitor()
    result = monitor.run_drift_check(force=True)

    print("\n" + "=" * 80)
    print("DRIFT CHECK RESULTS")
    print("=" * 80)
    print(f"Check ID: {result.get('check_id')}")
    print(f"Drift Detected: {result.get('drift_detected')}")
    print(f"Drifted Tests: {result.get('drifted_count')}")
    print(f"Alerts Generated: {result.get('alerts_generated')}")

    if result.get("drift_detected"):
        print("\n🚨 DRIFT DETECTED!")
        print("\nDrifted Features:")
        for feature in result["report"]["drifted_features"]:
            print(f"  - {feature['feature']} ({feature['test']})")
            print(f"    Statistic: {feature['statistic']:.4f}")
            print(f"    P-value: {feature['p_value']:.4f}")
            if feature["reference_mean"] and feature["current_mean"]:
                pct = (
                    (feature["current_mean"] - feature["reference_mean"])
                    / feature["reference_mean"]
                    * 100
                )
                print(
                    f"    Mean: {feature['reference_mean']:.2f} → {feature['current_mean']:.2f} ({pct:+.1f}%)"
                )

    monitor.close()
