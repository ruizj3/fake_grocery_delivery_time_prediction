# Delivery Time Prediction ML Pipeline

A production-style ML pipeline for predicting delivery times, demonstrating feature store architecture, model training, serving, and **automated drift monitoring**.

**Features:**
- ✅ Works with your existing `grocery_delivery.db` SQLite database
- ✅ Batch predictions for confirmed orders
- ✅ Real-time prediction API
- ✅ **Automated drift detection with anomaly alerting**
- ✅ Complete monitoring and retraining workflow

## Quick Reference

```bash
# Diagnose model performance issues
python diagnose_model_performance.py

# Validate training improvements
python validate_improvements.py

# Train model (with automatic feature materialization)
python train.py

# Run batch predictions on confirmed orders → stores in SQLite
python batch_predict.py

# Analyze prediction accuracy
python analyze_predictions.py

# Start real-time API
python -m delivery_ml.serving.api

# Check for drift (automated monitoring)
python check_drift.py

# Run continuous drift monitoring
python scheduled_drift_monitor.py
```

**What This Pipeline Does:**
- 🎯 Trains ML models to predict delivery times
- 📊 Runs batch predictions on confirmed orders
- 🚀 Provides real-time prediction API
- 🔍 **Automatically monitors for data drift**
- 🚨 **Alerts when model retraining is needed**
- 📈 Tracks prediction accuracy over time

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
│   │   ├── drift.py               # Drift detection (KS-test, PSI)
│   │   └── auto_drift_monitor.py  # Automated drift monitoring
│   ├── cli/
│   │   ├── train.py           # Training CLI
│   │   ├── batch_predict.py   # Batch prediction CLI
│   │   ├── analyze_predictions.py  # Prediction analysis CLI
│   │   └── check_drift.py     # Drift monitoring CLI
│   └── data/
│       └── schema_adapter.py  # Discover and adapt your schema
├── train.py                   # Convenience wrapper for training
├── batch_predict.py           # Convenience wrapper for batch predictions
├── analyze_predictions.py     # Convenience wrapper for analysis
├── check_drift.py             # Convenience wrapper for drift checking
├── scheduled_drift_monitor.py # Continuous drift monitoring daemon
├── examples/
│   ├── drift_monitoring_example.py   # Drift monitoring workflow
│   ├── batch_prediction_example.py   # Batch prediction example
│   └── integration_example.py        # Integration example
└── tests/
    ├── test_drift_system.py   # Drift detection tests
    ├── test_batch_api.py      # Batch API tests
    └── test_features.py       # Feature computation tests
```

**Note:** All CLI scripts in the root directory are convenience wrappers. The actual implementation is in `src/delivery_ml/cli/`. You can call them either way:

```bash
# Using root-level convenience wrappers
python train.py
python batch_predict.py
python check_drift.py

# Or calling the modules directly
python -m delivery_ml.cli.train
python -m delivery_ml.cli.batch_predict
python -m delivery_ml.cli.check_drift
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

```bash
python train.py
```

**What happens during training:**
- ✅ Automatically materializes aggregated features (restaurant & customer history)
- ✅ Uses enhanced XGBoost hyperparameters for better performance
- ✅ Expected R² score: 0.4-0.7 (vs 0.1 with basic parameters)
- ✅ Features: distance, time-of-day, order complexity, historical performance

**Performance expectations:**
- Training time: 2-5 minutes (depending on data size)
- Model metrics logged to MLflow
- Model saved to `models/delivery_time_model_latest.pkl`

### Multi-Model Training

Beyond the default XGBoost pipeline, the multi-model pipeline supports six model types:

| Model | Type | Key Strengths |
|-------|------|----------------|
| LightGBM | Boosting (recommended) | Fast, accurate, handles large datasets |
| Random Forest | Bagging | Robust, low variance |
| CatBoost | Boosting | Handles categoricals natively |
| Gradient Boosting | Boosting | Interpretable, sklearn-native |
| Extra Trees | Bagging | Fast training, reduced variance |
| Ridge | Linear | Fast, interpretable baseline |

```bash
# Train with a specific model type
python train_multi_model.py --model lightgbm
python train_multi_model.py --model random_forest
python train_multi_model.py --model catboost
python train_multi_model.py --model extra_trees
```

**Programmatic usage:**

```python
from delivery_ml.training.multi_model_pipeline import train_model

# Train a LightGBM model (default)
train_model("lightgbm")

# Train an Extra Trees model
train_model("extra_trees")

# Train with custom parameters
train_model("random_forest", n_estimators=500, max_depth=30)
```

### Hyperparameter Tuning

All models support hyperparameter tuning via `RandomizedSearchCV` (fast, recommended) or `GridSearchCV` (exhaustive). Each model has built-in parameter search spaces.

```bash
# Tune a single model with RandomizedSearchCV (30 iterations, 5-fold CV)
python train_multi_model.py --model lightgbm --tune

# Tune and train ALL models, then compare results
python train_multi_model.py --model all --tune

# Use exhaustive grid search instead
python train_multi_model.py --model all --tune --search-type grid

# Increase iterations for more thorough random search
python train_multi_model.py --model all --tune --n-iter 50

# Reduce cross-validation folds for faster tuning
python train_multi_model.py --model all --tune --cv 3

# Change the scoring metric
python train_multi_model.py --model all --tune --scoring neg_root_mean_squared_error
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tune` | off | Enable hyperparameter tuning before training |
| `--search-type` | `random` | `random` (fast) or `grid` (exhaustive) |
| `--n-iter` | `30` | Number of random combinations to try |
| `--cv` | `5` | Cross-validation folds |
| `--scoring` | `neg_mean_absolute_error` | Sklearn scoring metric |

**Programmatic usage:**

```python
from delivery_ml.training.multi_model_pipeline import train_model, MultiModelTrainer

# Train with automatic tuning (RandomizedSearchCV, 30 iterations, 5-fold CV)
train_model("lightgbm", tune=True)

# Customize the tuning process
train_model("random_forest", tune=True, tune_kwargs={
    "search_type": "grid",    # Exhaustive grid search
    "cv": 3,                  # 3-fold cross-validation
    "n_iter": 50,             # Iterations for randomized search
    "scoring": "neg_mean_absolute_error",
})
```

**Standalone tuning** — inspect results before committing to a full train:

```python
from delivery_ml.training.multi_model_pipeline import MultiModelTrainer

trainer = MultiModelTrainer(model_type="lightgbm")

# Run tuning separately
results = trainer.tune_hyperparameters(
    features_df,
    search_type="random",
    n_iter=50,
    cv=5,
)
print(results["best_params"])

# Then train with the best parameters
trainer.train(features_df, **results["best_params"])
```

**Custom parameter grid:**

```python
# Override the built-in search space with your own
custom_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.05],
}

results = trainer.tune_hyperparameters(
    features_df,
    search_type="grid",
    param_grid=custom_grid,
)
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

### Monitor for Drift (Automated)

The system now includes **automated drift detection** that runs on schedule and alerts you to anomalies:

```bash
# Run drift check manually
python check_drift.py

# View drift monitoring status
python check_drift.py --status

# View drift history
python check_drift.py --history

# Start continuous monitoring (runs every 24 hours)
python scheduled_drift_monitor.py

# Custom check interval (every 12 hours)
python scheduled_drift_monitor.py --interval 12
```

**Programmatic usage:**

```python
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor

# Initialize monitor
monitor = AutoDriftMonitor(
    reference_window_days=30,    # Historical baseline
    current_window_days=7,       # Recent comparison
    check_interval_hours=24,     # How often to check
    alert_on_n_features=3,       # Alert threshold
)

# Run drift check
result = monitor.run_drift_check(force=True)

if result['drift_detected']:
    print(f"🚨 Drift detected! {result['alerts_generated']} alerts")
    
    # Get unacknowledged alerts
    alerts = monitor.get_unacknowledged_alerts()
    for alert in alerts:
        print(f"  - {alert['feature']}: {alert['message']}")
    
    # Trigger retraining if critical drift
    if result['drifted_count'] >= 3:
        from delivery_ml.training.pipeline import train_model
        train_model(
            train_start=datetime.now() - timedelta(days=30),
            train_end=datetime.now(),
        )

monitor.close()
```

**See [DRIFT_MONITORING.md](DRIFT_MONITORING.md) for complete documentation.**

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
2. **Online Store (Redis)**: Optional caching for low-late

### Drift Monitoring CLI

| Command | Description |
|---------|-------------|
| `python check_drift.py` | Run drift check (respects interval) |
| `python check_drift.py --force` | Force drift check now |
| `python check_drift.py --status` | Show monitoring status |
| `python check_drift.py --history` | Show drift history (last 30 days) |
| `python check_drift.py --alerts` | Show unacknowledged alerts |
| `python check_drift.py --ack ID` | Acknowledge alert |
| `python scheduled_drift_monitor.py` | Run continuous monitoring (24h interval) |
| `python scheduled_drift_monitor.py --interval N` | Custom interval (N hours) |
| `python scheduled_drift_monitor.py --once` | Run once and exit |ncy serving

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
Drift Monitoring Database (drift_monitoring.db - local):**
| Table | Purpose |
|-------|---------|
| `drift_checks` | Drift check history |
| `drift_results` | Individual feature drift test results |
| `drift_alerts` | Generated alerts for drifted features |

**Feature Store Database (grocery_delivery.db):**| `stores` | Store locations (read-only) |
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
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor

# 1. Run batch predictions
predictor = BatchPredictor()
predictor.run_batch_predictions(save_to_db=True)
predictor.close()

# 2. Check for drift
monitor = AutoDriftMonitor()
result = monitor.run_drift_check(force=True)

if result['drift_detected'] and result['drifted_count'] >= 3:
    print("🚨 Critical drift detected - retraining model...")
    
    # Retrain on last 30 days of data
    train_model(
        train_start=datetime.now() - timedelta(days=30),
        train_end=datetime.now(),
    )
    
    print("✓ Model retrained successfully!")

monitor.close()
```

---

## Automated Drift Monitoring

### Overview

The system includes **production-ready drift detection** that automatically monitors for data anomalies and alerts when your model needs retraining.

**Key Features:**
- ✅ **Scheduled drift checks** (default: every 24 hours)
- ✅ **Statistical drift detection** (KS-test + PSI)
- ✅ **Severity-based alerts** (warning/critical)
- ✅ **Historical tracking** in SQLite database
- ✅ **CLI tools** for manual checks and status
- ✅ **Ready for email/Slack** notifications

### Quick Start - Drift Monitoring

```bash
# Run a drift check now
python check_drift.py

# View monitoring status
python check_drift.py --status

# View drift history (last 30 days)
python check_drift.py --history

# Show unacknowledged alerts
python check_drift.py --alerts

# Start continuous monitoring (runs every 24 hours)
python scheduled_drift_monitor.py

# Custom check interval (every 12 hours)
python scheduled_drift_monitor.py --interval 12
```

### How Drift Detection Works

The drift monitoring system compares two time windows of data:

1. **Reference Period** (Historical Baseline)
   - Default: Last 30 days of data (ending 7 days ago)
   - Represents what the model was trained on
   
2. **Current Period** (Recent Data)
   - Default: Last 7 days of data
   - Represents current production traffic

**Statistical Tests:**
- **KS-Test (Kolmogorov-Smirnov)**: Detects any distribution changes
  - P-value < 0.05 indicates significant drift
- **PSI (Population Stability Index)**: Industry standard for stability
  - PSI > 0.2 indicates significant drift

**Alert Levels:**
- **Warning**: 1-2 features drifted → Monitor
- **Critical**: 3+ features drifted → Retrain recommended

### Example: Drift Detection Output

```
================================================================================
DRIFT CHECK RESULTS
================================================================================
Check ID: 42
Drift Detected: True
Drifted Tests: 4
Alerts Generated: 4

🚨 DRIFT DETECTED!

Drifted Features:
  - haversine_distance (ks_test)
    Statistic: 0.1234
    P-value: 0.0023
    Mean: 5.23 → 6.87 (+31.4%)

  - total (psi)
    Statistic: 0.2456
    P-value: 0.0001
    Mean: 2450.00 → 2890.00 (+18.0%)

Recommendation: Consider retraining your model!
```

### Programmatic Usage

```python
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor

# Initialize monitor
monitor = AutoDriftMonitor(
    reference_window_days=30,    # Historical baseline
    current_window_days=7,       # Recent comparison
    check_interval_hours=24,     # How often to check
    alert_on_n_features=3,       # Alert threshold
    p_value_threshold=0.05,      # Statistical significance
    psi_threshold=0.2,           # PSI drift threshold
)

# Run drift check
result = monitor.run_drift_check(force=True)

if result['drift_detected']:
    print(f"🚨 Drift detected! {result['alerts_generated']} alerts")
    
    # Get unacknowledged alerts
    alerts = monitor.get_unacknowledged_alerts()
    for alert in alerts:
        print(f"  - {alert['feature']}: {alert['message']}")
    
    # Trigger retraining if critical drift
    if result['drifted_count'] >= 3:
        from delivery_ml.training.pipeline import train_model
        train_model(
            train_start=datetime.now() - timedelta(days=30),
            train_end=datetime.now(),
        )

# Get monitoring summary
summary = monitor.get_drift_summary()
print(f"Total checks: {summary['total_checks']}")
print(f"Drift rate: {summary['drift_rate']:.1%}")
print(f"Unacknowledged alerts: {summary['unacknowledged_alerts']}")

monitor.close()
```

### Drift Monitoring Database

The system maintains a `drift_monitoring.db` SQLite database with three tables:

#### drift_checks
Stores summary of each drift check run.

| Column | Description |
|--------|-------------|
| check_id | Unique identifier |
| check_timestamp | When check was run |
| reference_start/end | Reference data window |
| current_start/end | Current data window |
| drift_detected | Boolean flag |
| drifted_features_count | Number of features that drifted |
| report_json | Full drift report (JSON) |

#### drift_results
Stores individual feature test results.

| Column | Description |
|--------|-------------|
| result_id | Unique identifier |
| check_id | Foreign key to drift_checks |
| feature_name | Name of feature tested |
| test_name | Type of test (ks_test, psi) |
| statistic | Test statistic value |
| p_value | Statistical significance |
| is_drifted | Boolean flag |
| reference_mean | Mean in reference period |
| current_mean | Mean in current period |

#### drift_alerts
Stores generated alerts for drifted features.

| Column | Description |
|--------|-------------|
| alert_id | Unique identifier |
| check_id | Foreign key to drift_checks |
| severity | warning or critical |
| feature_name | Drifted feature |
| message | Human-readable description |
| drift_percentage | % change in mean |
| acknowledged | Boolean flag |

### Query Drift History

```python
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor

monitor = AutoDriftMonitor()

# Get drift history (last 60 days)
history = monitor.get_drift_history(days=60)
print(history)

# Get summary statistics
summary = monitor.get_drift_summary()
print(f"Drift rate: {summary['drift_rate']:.1%}")
print(f"Latest check: {summary['latest_check']}")
print(f"Next check due: {summary['next_check_due']}")

monitor.close()
```

### Continuous Monitoring

Run drift monitoring as a background service:

```bash
# Run continuously with default 24-hour interval
python scheduled_drift_monitor.py

# Custom interval (check every 6 hours)
python scheduled_drift_monitor.py --interval 6

# Run once and exit
python scheduled_drift_monitor.py --once

# Change log level
python scheduled_drift_monitor.py --log-level DEBUG
```

The daemon will:
1. Run drift checks on schedule
2. Generate and log alerts
3. Store results in database
4. Continue running until stopped (Ctrl+C for graceful shutdown)

### Running as a Service

#### Using systemd (Linux)

Create `/etc/systemd/system/drift-monitor.service`:

```ini
[Unit]
Description=ML Drift Monitoring Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/path/to/venv/bin/python scheduled_drift_monitor.py --interval 24
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable drift-monitor
sudo systemctl start drift-monitor
sudo systemctl status drift-monitor
```

#### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt && pip install -e .

CMD ["python", "scheduled_drift_monitor.py", "--interval", "24"]
```

Run:
```bash
docker build -t drift-monitor .
docker run -d --name drift-monitor \
  -v $(pwd)/drift_monitoring.db:/app/drift_monitoring.db \
  -v $(pwd)/grocery_delivery.db:/app/grocery_delivery.db \
  drift-monitor
```

### Integration with Retraining

When drift is detected, automatically trigger retraining:

```python
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor
from delivery_ml.training.pipeline import train_model
from datetime import datetime, timedelta

monitor = AutoDriftMonitor()

# Check for drift
result = monitor.run_drift_check(force=True)

if result['drift_detected']:
    print(f"Drift detected in {result['drifted_count']} tests")
    
    # Trigger retraining on critical drift
    if result['drifted_count'] >= 3:
        print("Critical drift - triggering model retraining...")
        
        # Retrain on last 30 days
        train_model(
            train_start=datetime.now() - timedelta(days=30),
            train_end=datetime.now(),
        )
        
        print("✓ Model retrained successfully!")
        
        # Acknowledge all alerts
        alerts = monitor.get_unacknowledged_alerts()
        for alert in alerts:
            monitor.acknowledge_alert(alert['alert_id'])

monitor.close()
```

### Configuration Options

All drift monitoring settings can be customized:

```python
monitor = AutoDriftMonitor(
    # Database path
    db_path=Path("drift_monitoring.db"),
    
    # Time windows
    reference_window_days=30,    # Historical baseline
    current_window_days=7,       # Recent comparison
    
    # Check frequency
    check_interval_hours=24,     # How often to check
    
    # Statistical thresholds
    p_value_threshold=0.05,      # KS-test threshold
    psi_threshold=0.2,           # PSI threshold
    
    # Alert threshold
    alert_on_n_features=3,       # Alert if N+ features drift
)
```

### Best Practices

1. **Set appropriate time windows**
   - Reference: 30+ days for stable baseline
   - Current: 7 days to catch recent changes
   - Avoid overlapping windows

2. **Check frequency**
   - Daily checks for production models
   - More frequent (6-12 hours) for critical applications
   - Less frequent (weekly) for stable systems

3. **Alert thresholds**
   - Start with 3+ features for critical alerts
   - Adjust based on false positive rate
   - Lower for high-stakes applications

4. **Response protocol**
   - Investigate all critical alerts
   - Retrain model when drift confirmed
   - Document drift patterns
   - Track seasonal variations

5. **Monitor drift trends**
   - Track drift rate over time
   - Look for seasonal patterns
   - Correlate with business changes
   - Set up dashboards for visualization

### Troubleshooting

**No drift detected when expected:**
- Check if enough data in current window
- Verify reference period is representative
- Lower p_value_threshold (e.g., 0.10)
- Check feature computation logic

**Too many false positives:**
- Increase reference window (e.g., 60 days)
- Raise p_value_threshold (e.g., 0.01)
- Increase alert_on_n_features threshold
- Consider seasonal adjustments

**Insufficient data errors:**
- Ensure database has enough historical data
- Check date filters in queries
- Verify orders have delivery timestamps
- Check for data quality issues

### Example Workflow

Complete drift monitoring and retraining workflow:

```bash
# 1. Train initial model
python train.py

# 2. Start prediction API
python -m delivery_ml.serving.api &

# 3. Run batch predictions
python batch_predict.py

# 4. Start continuous drift monitoring
python scheduled_drift_monitor.py &

# 5. Check drift status (anytime)
python check_drift.py --status

# 6. View drift history
python check_drift.py --history

# 7. If drift detected, retrain model
# This happens automatically in the workflow above,
# or manually trigger:
python train.py
```

**Python workflow:**

```python
# examples/drift_monitoring_example.py
from datetime import datetime, timedelta
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor
from delivery_ml.training.pipeline import train_model

monitor = AutoDriftMonitor()

# Run drift check
result = monitor.run_drift_check(force=True)

if result['drift_detected']:
    alerts = monitor.get_unacknowledged_alerts()
    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
    
    if critical_alerts:
        # Critical drift - retrain model
        train_model(
            train_start=datetime.now() - timedelta(days=30),
            train_end=datetime.now(),
        )
        
        # Acknowledge alerts
        for alert in alerts:
            monitor.acknowledge_alert(alert['alert_id'])

monitor.close()
```

### Drift Monitoring CLI Reference

| Command | Description |
|---------|-------------|
| `python check_drift.py` | Run drift check (respects interval) |
| `python check_drift.py --force` | Force drift check now |
| `python check_drift.py --status` | Show monitoring status |
| `python check_drift.py --history` | Show drift history (last 30 days) |
| `python check_drift.py --history --days 60` | Show 60 days of history |
| `python check_drift.py --alerts` | Show unacknowledged alerts |
| `python check_drift.py --ack ID` | Acknowledge alert |
| `python scheduled_drift_monitor.py` | Run continuous monitoring (24h interval) |
| `python scheduled_drift_monitor.py --interval N` | Custom interval (N hours) |
| `python scheduled_drift_monitor.py --once` | Run once and exit |
| `python test_drift_system.py` | Run drift detection tests |
| `python examples/drift_monitoring_example.py` | Full workflow example |

---

## Complete Workflow

### End-to-End Production Pipeline

```bash
# 1. Setup
python -m venv delivery_prediction_venv
source delivery_prediction_venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Train initial model
python train.py

# 3. Start services
python -m delivery_ml.serving.api &           # Prediction API
python scheduled_drift_monitor.py &           # Drift monitoring

# 4. Run batch predictions (scheduled or on-demand)
python batch_predict.py

# 5. Monitor system health
python check_drift.py --status               # Drift status
python analyze_predictions.py                # Prediction accuracy

# 6. If drift detected, retrain
python train.py
```

## Next Steps & Extensions

### Implemented ✅
1. ✅ **Automated drift detection** - Complete with alerting and history tracking
2. ✅ **Batch prediction pipeline** - Process confirmed orders automatically
3. ✅ **Real-time API** - FastAPI with batch and single predictions
4. ✅ **Feature store** - SQLite offline + optional Redis online
5. ✅ **Prediction tracking** - Store and analyze prediction accuracy

### Suggested Enhancements
1. **Email/Slack notifications** for drift alerts
   - Implement `_send_email_alert()` in `auto_drift_monitor.py`
   - Configure SMTP or Slack webhook
   
2. **Shadow deployment** to compare model versions
   - Run multiple model versions in parallel
   - Compare predictions before promoting
   
3. **Prometheus metrics** for production monitoring
   - Export drift metrics
   - Track API latency and throughput
   - Alert on SLO violations
   
4. **Automated retraining pipeline** triggered by drift
   - GitHub Actions or Airflow DAG
   - Auto-deploy new models on success
   
5. **Slice-based evaluation** 
   - Performance by restaurant, time of day, distance
   - Identify model weaknesses
   
6. **Drift visualization dashboard** with Streamlit
   - Interactive drift history plots
   - Feature importance tracking
   - Alert management UI

## Summary

This ML pipeline provides a **production-ready** delivery time prediction system with:

✅ **Training**: XGBoost + multi-model pipeline (LightGBM, Random Forest, CatBoost, Gradient Boosting, Extra Trees, Ridge) with MLflow tracking  
✅ **Tuning**: Built-in hyperparameter tuning via GridSearchCV / RandomizedSearchCV for all models  
✅ **Serving**: FastAPI for real-time and batch predictions  
✅ **Monitoring**: Automated drift detection with alerting  
✅ **Storage**: SQLite for features, predictions, and drift history  
✅ **Tooling**: Organized CLI tools in `src/delivery_ml/cli/`  
✅ **Testing**: Comprehensive test suite in `tests/`  

### Running Tests

```bash
# Run all tests with pytest
python -m pytest tests/ -v

# Run specific test files
python tests/test_drift_system.py
python tests/test_batch_api.py
python tests/test_features.py

# Run with coverage
python -m pytest tests/ --cov=delivery_ml --cov-report=html
```

The system is fully operational and tested. All drift detection tests pass, and the monitoring system is ready to catch data quality issues before they affect your model's performance.

**Get started now:**
```bash
python test_drift_system.py              # Verify installation
python check_drift.py                    # Check for drift
python examples/drift_monitoring_example.py  # See full workflow
```

For questions or issues, check the inline documentation and example scripts!
