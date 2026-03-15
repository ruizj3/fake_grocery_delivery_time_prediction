#!/usr/bin/env python3
"""Run batch predictions on confirmed orders.

This is a convenience wrapper for the main batch prediction CLI.
"""

if __name__ == "__main__":
    from delivery_ml.cli.batch_predict import main
    main()
