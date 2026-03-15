#!/usr/bin/env python3
"""Check for drift in model features.

This is a convenience wrapper for the main drift checking CLI.
"""

if __name__ == "__main__":
    from delivery_ml.cli.check_drift import main
    main()
