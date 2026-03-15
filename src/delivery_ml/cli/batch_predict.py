#!/usr/bin/env python
"""
Batch Prediction CLI

Run predictions on confirmed orders and store results in the database.

Usage:
    # Process all confirmed orders
    python batch_predict.py
    
    # Process only 10 orders
    python batch_predict.py --limit 10
    
    # Show prediction statistics
    python batch_predict.py --stats
"""

from delivery_ml.serving.batch_predictor import main

if __name__ == "__main__":
    main()
