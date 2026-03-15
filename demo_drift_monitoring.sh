#!/bin/bash
# Drift Monitoring Demo Script
# This script demonstrates the drift monitoring capabilities

echo "=========================================="
echo "DRIFT MONITORING DEMONSTRATION"
echo "=========================================="
echo ""

# Step 1: Show current status
echo "Step 1: Checking current drift monitoring status..."
python check_drift.py --status
echo ""

# Step 2: Run the test to verify system works
echo "Step 2: Running drift detection tests..."
python -m pytest tests/test_drift_system.py -v || python tests/test_drift_system.py
echo ""

# Step 3: Show the example
echo "Step 3: Running drift monitoring example..."
echo "(This will show how drift detection integrates with retraining)"
python examples/drift_monitoring_example.py
echo ""

# Step 4: Show available commands
echo "Step 4: Available drift monitoring commands:"
echo ""
echo "  python check_drift.py           - Run drift check"
echo "  python check_drift.py --status  - Show status"
echo "  python check_drift.py --history - Show history"
echo "  python check_drift.py --alerts  - Show alerts"
echo ""
echo "  python scheduled_drift_monitor.py  - Continuous monitoring"
echo ""

echo "=========================================="
echo "DEMO COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check the README.md for complete documentation"
echo "2. Run 'python check_drift.py --status' to monitor drift"
echo "3. Run 'python train.py' to retrain the model"
echo ""
