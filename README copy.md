# Delivery Time Prediction ML Pipeline

A production-style ML pipeline for predicting delivery times, demonstrating feature store architecture, model training, serving, and monitoring.

**Updated to work with your existing `grocery_delivery.db` SQLite database.**

## Quick Reference

```bash
# Train model
python train.py

# Run batch predictions on confirmed orders → stores in SQLite
python batch_predict.py

# Analyze prediction accuracy
python analyze_predictions.py

# Start real-time API
python -m delivery_ml.serving.api
```

**New Feature:** Batch predictions automatically retrieve confirmed orders from your database, generate predictions, and store them in the `ml_predictions` table!

## Project Structure

```
delivery_ml/
├── src/delivery_ml/
│   ├── config.py              # Pydantic settings
│   ├── schemas.py             # Data schemas and validation
│   ├── features/
│   │   ├── definitions.py     # Feature definitions (WHAT)
│   │   ├── computation.py     # Feature computation (HOW) - SQLite
│   │   └── store.py           # Offline (SQLite) + Online (Redis) stores
│   ├── training/
│   │   └── pipeline.py        # XGBoost training with MLflow
│   ├── serving/
│   │   ├── api.py             # FastAPI real-time prediction endpoint
│   │   └── batch_predictor.py # Batch predictions for confirmed orders
│   ├── monitoring/
│   │   └── drift.py           # Drift detection
│   └── data/
│       └── schema_adapter.py  # Discover and adapt your schema
├── batch_predict.py           # CLI for batch predictions
└── tests/
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv delivery_prediction_venv
source delivery_prediction_venv/bin/activate

# Install package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Discover Your Database Schema

First, understand what's in your `grocery_delivery.db`:

```bash
python -m delivery_ml.data.schema_adapter
```

This will:
- Show all tables and columns in your database
- Check compatibility with expected schema
- Suggest column mappings if needed

### 3. Configure Database Path

Create a `.env` file in the project root with your database path:

```bash
DELIVERY_ML_SQLITE_DB_PATH=/path/to/your/grocery_delivery.db
DELIVERY_ML_API_PORT=8000
```

The pipeline works with a multi-table schema and automatically handles JOINs:

**Required Tables:**
- `orders` - Order information (order_id, customer_id, store_id, created_at, confirmed_at, delivered_at, delivery_latitude, delivery_longitude, total)
- `stores` - Store locations (store_id, latitude, longitude)
- `order_items` - Items per order (order_id, quantity) - optional, defaults to 1 if missing

The batch predictor automatically:
- JOINs `orders` with `stores` to get store coordinates
- Aggregates `order_items` to get total quantity per order
- Filters for confirmed orders: `WHERE confirmed_at IS NOT NULL AND delivered_at IS NULL`

### Predictions Table Schema

The batch predictor automatically creates an `ml_predictions` table to store prediction results:

| Column | Type | Description |
|--------|------|-------------|
| `prediction_id` | INTEGER | Auto-incrementing primary key |
| `order_id` | TEXT | Reference to order |
| `customer_id` | TEXT | Customer identifier |
| `store_id` | TEXT | Restaurant/store identifier |
| `predicted_delivery_minutes` | REAL | Model prediction |
| `prediction_timestamp` | TEXT | When prediction was made (ISO datetime) |
| `model_version` | TEXT | Version of model used |
| `features_json` | TEXT | JSON with features used for prediction |
| `created_at` | TEXT | Database insert timestamp |

Unique constraint on `(order_id, model_version)` ensures no duplicate predictions per model version.

### 4. Train a Model

```python
python train.py
```

### 5. Run Batch Predictions on Confirmed Orders

Process all confirmed orders (orders without `delivered_at` set) and store predictions in the database:

```bash
# Process all confirmed orders
python batch_predict.py

# Process only 10 orders
python batch_predict.py --limit 10

# Show prediction statistics
python batch_predict.py --stats
```

This will:
- Retrieve all confirmed orders from the `orders` table (in source database)
- Generate predictions for each order
- Store results in the local `predictions.db` with:
  - `order_id`, `customer_id`, `store_id`
  - `predicted_delivery_minutes`
  - `model_version`, `features_json`
  - `prediction_timestamp`, `created_at`

### 6. Start the Real-Time Prediction API

```bash
python -m delivery_ml.serving.api
```

The API will be available at `http://localhost:3000`.

**The API now supports two modes:**

1. **Real-time single predictions** - For ad-hoc prediction requests
2. **Batch predictions from your service** - Receives confirmed orders and stores predictions

### 7. Send Confirmed Orders from Your Service

When orders are confirmed in your other service, send them to the batch prediction endpoint:

```python
import requests
from datetime import datetime

# In your grocery delivery service, when order is confirmed:
confirmed_orders = {
    "orders": [
        {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "store_id": order.store_id,
            "store_latitude": store.latitude,
            "store_longitude": store.longitude,
            "delivery_latitude": order.delivery_latitude,
            "delivery_longitude": order.delivery_longitude,
            "total": order.total,
            "quantity": total_items,
            "created_at": order.created_at.isoformat(),
        }
    ]
}

# Send to prediction API
response = requests.post(
    "http://localhost:3000/predict/batch",
    json=confirmed_orders
)

predictions = response.json()
# predictions are automatically saved to predictions.db
```

You can also test this with:
```bash
python test_batch_api.py
```

### 8. Make Real-Time Predictions (Ad-hoc)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "test_001",
    "customer_id": "cust_abc123",
    "store_id": "rest_xyz789",
    "delivery_latitude": 47.65,
    "delivery_longitude": -122.35,
    "latitude": 47.60,
    "longitude": -122.33,
    "total": 2500,
    "quantity": 3
  }'
```

## Configuration

**Important:** Create a `.env` file in the project root to configure your database path:

```bash
# .env file
DELIVERY_ML_SQLITE_DB_PATH=/Users/you/path/to/grocery_delivery.db
DELIVERY_ML_API_PORT=8000
```

**Note:** The system uses two separate databases:
- **Source database** (grocery_delivery.db): Reads orders, stores, and other source data from your existing database
- **Predictions database** (predictions.db): Stores ML predictions locally in this repo

This separation keeps your source data clean and allows you to manage predictions independently.

All settings can be configured via environment variables (prefix with `DELIVERY_ML_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SQLITE_DB_PATH` | `grocery_delivery.db` | **Path to your SQLite database** |
| `MODEL_DIR` | `models` | Model storage directory |
| `REDIS_HOST` | `localhost` | Redis host (optional) |
| `REDIS_PORT` | `6379` | Redis port |
| `API_PORT` | `8000` | API server port |

Example:
```bash
export DELIVERY_ML_SQLITE_DB_PATH=/path/to/grocery_delivery.db
```

## Working with Live Data

Since your database is constantly updated with fake "live" data, you can:

### Run Batch Predictions Programmatically

```python
from delivery_ml.serving.batch_predictor import BatchPredictor

# Initialize predictor
predictor = BatchPredictor()

# Process all confirmed orders
predictions_df = predictor.run_batch_predictions(
    limit=100,  # Process only 100 orders
    save_to_db=True,  # Save to ml_predictions table
)

# Get prediction statistics
stats = predictor.get_prediction_stats()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Average prediction: {stats['avg_prediction_minutes']:.1f} minutes")

# Clean up
predictor.close()
```

### Query Stored Predictions

```python
import sqlite3
import polars as pl

# Query from local predictions database
predictions_conn = sqlite3.connect("predictions.db")

# Get recent predictions
predictions = pl.read_database(
    """
    SELECT 
        order_id,
        predicted_delivery_minutes,
        prediction_timestamp,
        model_version
    FROM ml_predictions
    ORDER BY prediction_timestamp DESC
    LIMIT 100
    """,
    connection=predictions_conn,
)

predictions_conn.close()

# To compare with actuals, join with source database
source_conn = sqlite3.connect("/path/to/grocery_delivery.db")
orders = pl.read_database(
    """
    SELECT order_id, created_at, delivered_at
    FROM orders
    WHERE delivered_at IS NOT NULL
    """,
    connection=source_conn,
)
source_conn.close()

# Join and calculate accuracy
combined = predictions.join(orders, on="order_id")
# ... calculate metrics
```

### Analyze Prediction Accuracy

Once some predicted orders have been delivered, analyze accuracy:

```bash
# Run accuracy analysis
python analyze_predictions.py
```

This will show:
- Mean Absolute Error (MAE) and RMSE
- Percentage of predictions within 5/10/15 minutes
- Best and worst predictions
- Performance breakdown by store
- Error distribution statistics

### Retrain on New Data

```python
from datetime import datetime, timedelta
from delivery_ml.training.pipeline import train_model

# Train on last 30 days
train_model(
    train_start=datetime.now() - timedelta(days=30),
    train_end=datetime.now(),
)
```

### Monitor for Drift

```python
from delivery_ml.monitoring.drift import DriftDetector
from delivery_ml.features.store import FeatureStore

store = FeatureStore()

# Get reference data (what model was trained on)
reference_df = store.offline.get_training_data(
    train_start, train_end
)

# Get recent data
recent_df = store.offline.get_training_data(
    datetime.now() - timedelta(days=7),
    datetime.now(),
)

# Check for drift
detector = DriftDetector()
results = detector.check_all_features(reference_df, recent_df)
report = detector.generate_report(results)

if report["drift_detected"]:
    print("⚠️ Drift detected! Consider retraining.")
    for feature in report["drifted_features"]:
        print(f"  - {feature['feature']}: {feature['test']}")
```

## Key Concepts

### Point-in-Time Correctness

When computing features for training, we only use data available at prediction time:

```python
# WRONG (data leakage):
"Average delivery time for restaurant X" using ALL historical data

# CORRECT:
"Average delivery time for restaurant X" using only orders
completed BEFORE the current order was placed
```

### Two-Layer Feature Store

1. **Offline Store (SQLite)**: Your `grocery_delivery.db` - source of truth
2. **Online Store (Redis)**: Optional caching for low-latency serving

Redis is optional. If not available, the system falls back to SQLite queries.

## API Endpoints

### Real-Time Prediction API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata and feature importance |
| `/predict` | POST | Single order prediction (ad-hoc) |
| `/predict/batch` | POST | **Batch predictions for confirmed orders** |
| `/predictions/recent` | GET | Get recent predictions from database |
| `/predictions/stats` | GET | Get prediction statistics |
| `/monitoring/stats` | GET | API request statistics |
| `/monitoring/requests` | GET | Recent API requests log |

### Batch Prediction CLI

| Command | Description |
|---------|-------------|
| `python batch_predict.py` | Process all confirmed orders |
| `python batch_predict.py --limit N` | Process only N orders |
| `python batch_predict.py --stats` | Show prediction statistics |
| `python analyze_predictions.py` | Analyze prediction accuracy |
| `python analyze_predictions.py --export` | Export predictions to CSV |
| `python test_batch_api.py` | Test batch prediction API |

### Database Tables

**Source Database (grocery_delivery.db):**
| Table | Purpose |
|-------|---------|
| `orders` | Order information (read-only) |
| `stores` | Store locations (read-only) |
| `order_items` | Order items (read-only) |
| `customers`, `drivers`, etc. | Other source tables |

**Predictions Database (predictions.db - local):**
| Table | Purpose |
|-------|---------|
| `ml_predictions` | Batch prediction results |

**Feature Store Database (grocery_delivery.db):**
| Table | Purpose |
|-------|---------ediction accuracy |

### Database Tables

| Table | Purpose |
|-------|---------|
| `orders` | Source data (your existing table) |
| `ml_predictions` | Stored batch predictions |
| `ml_restaurant_features` | Materialized restaurant features |
| `ml_customer_features` | Materialized customer features |
| `ml_feature_metadata` | Feature versioning metadata |

## Adapting to Your Schema

If your database has different column names:

```bash
# Generate a mapping template
python -m delivery_ml.data.schema_adapter --generate-template

# Edit column_mapping.json to map your columns

# Apply the mapping (creates a view)
python -m delivery_ml.data.schema_adapter --apply-mapping column_mapping.json
```

Then update `settings.sqlite_db_path` or modify queries to use `orders_ml_view` instead of `orders`.

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## Complete Workflow Example

Here's a typical end-to-end workflow:

```bash
# 1. Check your database schema
python -m delivery_ml.data.schema_adapter

# 2. Train a model on historical data
python train.py

# 3. Start the prediction API (runs in background, waiting for orders)
python -m delivery_ml.serving.api

# 4. In your other service, send confirmed orders to the API
#    (predictions are automatically saved to predictions.db)

# 5. (Optional) Run manual batch predictions on existing orders
python batch_predict.py

# 6. Check prediction statistics
python batch_predict.py --stats

# 7. Wait for some orders to be delivered, then analyze accuracy
python analyze_predictions.py

# 8. Export predictions to CSV
python analyze_predictions.py --export
```

### Integration with Your Grocery Delivery Service

In your other service, when an order is confirmed, send it to this API:

```python
# In your grocery_delivery_service code, after order confirmation:

import requests

def send_order_for_prediction(order, store):
    """Send confirmed order to prediction API."""
    
    # Aggregate quantity from order items
    total_quantity = sum(item.quantity for item in order.items)
    
    prediction_request = {
        "orders": [{
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "store_id": order.store_id,
            "store_latitude": store.latitude,
            "store_longitude": store.longitude,
            "delivery_latitude": order.delivery_latitude,
            "delivery_longitude": order.delivery_longitude,
            "total": order.total,
            "quantity": total_quantity,
            "created_at": order.created_at.isoformat(),
        }]
    }
    
    try:
        response = requests.post(
            "http://localhost:3000/predict/batch",
            json=prediction_request,
            timeout=5
        )
        response.raise_for_status()
        
        result = response.json()
        if result['successful'] > 0:
            prediction = result['predictions'][0]
            print(f"Predicted delivery time: {prediction['predicted_delivery_minutes']:.1f} min")
            # Optionally store prediction_id in your database
            return prediction
    except Exception as e:
        print(f"Prediction API error: {e}")
        # Continue without prediction - it's not critical
    
    return None

# Call this when order is confirmed
def on_order_confirmed(order_id):
    order = get_order(order_id)
    store = get_store(order.store_id)
    
    # Send to prediction API (non-blocking, fire-and-forget style)
    prediction = send_order_for_prediction(order, store)
    
    # Continue with your normal order processing
    # ...
```

### Monitoring and Retraining Loop

```python
from datetime import datetime, timedelta
from delivery_ml.serving.batch_predictor import BatchPredictor
from delivery_ml.training.pipeline import train_model

# 1. Run batch predictions
predictor = BatchPredictor()
predictor.run_batch_predictions(save_to_db=True)
predictor.close()

# 2. Check if retraining is needed (e.g., weekly)
# Train on last 30 days of data
train_model(
    train_start=datetime.now() - timedelta(days=30),
    train_end=datetime.now(),
)

# 3. New model is saved, batch predictor will use it automatically
# on next run
```

## Next Steps

Ideas for extending this project:

1. **Add scheduled retraining** with Dagster or Prefect
2. **Implement shadow deployment** to compare model versions
3. **Add Prometheus metrics** for production monitoring
4. **Build drift alerts** that trigger retraining
5. **Add slice-based evaluation** (by restaurant, time of day, distance)
