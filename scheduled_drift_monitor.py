#!/usr/bin/env python3
"""Scheduled drift monitoring daemon.

This script runs continuously and checks for drift at regular intervals.
Can be run as a background service or in a container.

Usage:
    python scheduled_drift_monitor.py                    # Run with defaults
    python scheduled_drift_monitor.py --interval 12      # Check every 12 hours
    python scheduled_drift_monitor.py --once             # Run once and exit
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime

from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def run_monitoring_loop(
    check_interval_hours: int = 24,
    run_once: bool = False,
):
    """Run continuous drift monitoring."""
    global shutdown_requested

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 80)
    logger.info("DRIFT MONITORING DAEMON STARTED")
    logger.info("=" * 80)
    logger.info(f"Check interval: {check_interval_hours} hours")
    logger.info(f"Mode: {'One-time' if run_once else 'Continuous'}")
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 80)

    monitor = AutoDriftMonitor(check_interval_hours=check_interval_hours)

    try:
        while not shutdown_requested:
            logger.info(f"\n[{datetime.now()}] Checking if drift check is due...")

            try:
                result = monitor.run_drift_check()

                if result.get("skipped"):
                    logger.info(f"Check skipped: {result.get('reason')}")
                elif result.get("error"):
                    logger.error(f"Check failed: {result.get('error')}")
                else:
                    logger.info(f"✓ Check complete - ID: {result.get('check_id')}")

                    if result.get("drift_detected"):
                        logger.warning(
                            f"🚨 DRIFT DETECTED! {result.get('alerts_generated')} alerts generated"
                        )
                        # In production, this would trigger notifications
                    else:
                        logger.info("✓ No drift detected")

            except Exception as e:
                logger.error(f"Error during drift check: {e}", exc_info=True)

            if run_once:
                logger.info("One-time check complete, exiting...")
                break

            # Sleep in small increments to allow for graceful shutdown
            sleep_duration = check_interval_hours * 3600  # Convert to seconds
            sleep_interval = 60  # Check for shutdown every minute

            for _ in range(int(sleep_duration / sleep_interval)):
                if shutdown_requested:
                    break
                time.sleep(sleep_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down drift monitor...")
        monitor.close()
        logger.info("Drift monitor stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Scheduled drift monitoring daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Check interval in hours (default: 24)",
    )

    parser.add_argument(
        "--once", action="store_true", help="Run once and exit (don't loop)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("drift_monitor.log"),
        ],
    )

    # Run monitoring
    run_monitoring_loop(
        check_interval_hours=args.interval,
        run_once=args.once,
    )


if __name__ == "__main__":
    main()
