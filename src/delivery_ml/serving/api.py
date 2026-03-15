"""FastAPI serving endpoint for delivery time predictions."""

import asyncio
import json
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from delivery_ml.config import settings
from delivery_ml.features.store import FeatureStore
from delivery_ml.monitoring.auto_drift_monitor import AutoDriftMonitor
from delivery_ml.schemas import PredictionRequest, PredictionResponse
from delivery_ml.training.pipeline import DeliveryTimeModel, train_model


# Global state
model: DeliveryTimeModel | None = None
feature_store: FeatureStore | None = None
predictions_db_path: Path = Path("predictions.db")
drift_monitor: AutoDriftMonitor | None = None
drift_monitor_thread: threading.Thread | None = None
stop_drift_monitor = threading.Event()
model_lock = threading.Lock()  # Protects model access during reload


def run_drift_monitoring_loop():
    """Background thread that monitors for drift and triggers retraining."""
    global model, drift_monitor
    cih = 1  # Check interval hours
    print(f"🔍 Drift monitoring started (checking every {cih} hours)")
    
    # Wait a bit for the API to fully start
    time.sleep(10)
    
    while not stop_drift_monitor.is_set():
        try:
            # Initialize monitor if needed
            if drift_monitor is None:
                drift_monitor = AutoDriftMonitor(
                    check_interval_hours=cih,  # Check every 6 hours
                    alert_on_n_features=3,   # Alert if 3+ features drift
                )
            
            # Run drift check
            print(f"[{datetime.now()}] Running drift check...")
            result = drift_monitor.run_drift_check()
            
            if result.get("error"):
                print(f"⚠️  Drift check error: {result['error']}")
            elif result.get("skipped"):
                print(f"⏭️  Drift check skipped: {result['reason']}")
            elif result.get("drift_detected"):
                drifted_count = result.get("drifted_count", 0)
                print(f"🚨 DRIFT DETECTED! {drifted_count} drifted tests")
                
                # Check if we should retrain (critical drift)
                # With 9 features, we have 18 total tests (9 × 2: KS + PSI)
                # Retrain if 33%+ of tests fail (6+ out of 18)
                if drifted_count >= 6:
                    print("🔄 Critical drift detected - triggering automatic retraining...")
                    
                    try:
                        # Retrain model on last 31 days with 20% test split
                        # This happens in the background without blocking predictions
                        print("Triggering automatic retraining on last 31 days of data...")
                        new_model_version = train_model()
                        
                        print(f"✅ Model retrained successfully! Version: {new_model_version}")
                        
                        # Reload the model with thread-safe swap
                        # Predictions continue using old model during this brief lock
                        model_path = settings.model_dir / "delivery_time_model_latest.pkl"
                        new_model = DeliveryTimeModel.load(model_path)
                        
                        with model_lock:
                            model = new_model
                        
                        print(f"✅ New model loaded into API: {model.version}")
                        
                        # Acknowledge alerts
                        alerts = drift_monitor.get_unacknowledged_alerts()
                        for alert in alerts:
                            drift_monitor.acknowledge_alert(alert['alert_id'])
                        print(f"✅ Acknowledged {len(alerts)} drift alerts")
                        
                    except Exception as e:
                        print(f"❌ Retraining failed: {e}")
                        print("   Predictions will continue with existing model")
                else:
                    print(f"⚠️  Warning-level drift detected (not critical yet)")
            else:
                print("✅ No drift detected - model is healthy")
        
        except Exception as e:
            print(f"❌ Error in drift monitoring loop: {e}")
        
        # Sleep for a bit before next check (check every 5 minutes if interval hasn't elapsed)
        for _ in range(60):  # Check every 5 seconds for stop signal
            if stop_drift_monitor.is_set():
                break
            time.sleep(5)
    
    print("🛑 Drift monitoring stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize feature store on startup."""
    global model, feature_store, predictions_db_path, drift_monitor_thread, stop_drift_monitor

    # Load model
    model_path = settings.model_dir / "delivery_time_model_latest.pkl"
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        with model_lock:
            model = None
    else:
        loaded_model = DeliveryTimeModel.load(model_path)
        with model_lock:
            model = loaded_model
        print(f"Loaded model version: {model.version}")

    # Initialize feature store
    feature_store = FeatureStore()
    feature_store.initialize()
    print("Feature store initialized")
    
    # Initialize predictions database
    _initialize_predictions_db()
    print(f"Predictions database initialized at {predictions_db_path}")
    
    # Start drift monitoring in background thread
    stop_drift_monitor.clear()
    drift_monitor_thread = threading.Thread(
        target=run_drift_monitoring_loop,
        daemon=True,
        name="DriftMonitor"
    )
    drift_monitor_thread.start()
    print("✅ Background drift monitoring thread started")

    yield

    # Cleanup
    print("Shutting down API...")
    stop_drift_monitor.set()
    
    if drift_monitor_thread and drift_monitor_thread.is_alive():
        drift_monitor_thread.join(timeout=10)
        print("Drift monitoring thread stopped")
    
    if drift_monitor:
        drift_monitor.close()
    
    if feature_store:
        feature_store.close()
    
    print("API shutdown complete")


app = FastAPI(
    title="Delivery Time Prediction API",
    description="Predicts delivery time for orders using ML",
    version="0.1.0",
    lifespan=lifespan,
)


# Request logging for monitoring
request_log: list[dict[str, Any]] = []


# Batch prediction schemas
class ConfirmedOrder(BaseModel):
    """Schema for confirmed order from external service."""
    order_id: str
    customer_id: str
    store_id: str
    store_latitude: float
    store_longitude: float
    delivery_latitude: float
    delivery_longitude: float
    total: int
    quantity: int | None = 1
    subtotal: float | None = 0.0
    delivery_fee: float | None = 0.0
    tip: float | None = 0.0
    item_count: int | None = 0
    traffic_multiplier: float | None = 1.0
    weather_condition: str | None = None
    is_peak_hour: bool | None = False
    created_at: str  # ISO datetime string


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions on confirmed orders."""
    orders: list[ConfirmedOrder]


class OrderPrediction(BaseModel):
    """Single order prediction result."""
    order_id: str
    predicted_delivery_minutes: float
    model_version: str
    prediction_timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[OrderPrediction]
    total_orders: int
    successful: int
    failed: int
    errors: list[dict[str, str]] = []


def _initialize_predictions_db() -> None:
    """Initialize predictions database table."""
    conn = sqlite3.connect(predictions_db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                store_id TEXT NOT NULL,
                predicted_delivery_minutes REAL NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(order_id, model_version)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_order_id 
            ON ml_predictions(order_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
            ON ml_predictions(prediction_timestamp)
        """)
        
        conn.commit()
    finally:
        conn.close()


def _save_prediction_to_db(
    order_id: str,
    customer_id: str,
    store_id: str,
    prediction: float,
    features: dict[str, Any],
    model_version: str,
    timestamp: datetime,
) -> None:
    """Save prediction to local database."""
    conn = sqlite3.connect(predictions_db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO ml_predictions (
                order_id,
                customer_id,
                store_id,
                predicted_delivery_minutes,
                prediction_timestamp,
                model_version,
                features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                customer_id,
                store_id,
                prediction,
                timestamp.isoformat(),
                model_version,
                json.dumps(features),
            ),
        )
        conn.commit()
    finally:
        conn.close()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    with model_lock:
        is_loaded = model is not None
        version = model.version if model else "none"
    
    return {
        "status": "healthy",
        "model_loaded": str(is_loaded),
        "model_version": version,
    }


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    """Get information about the loaded model."""
    with model_lock:
        current_model = model
    
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "version": current_model.version,
        "features": current_model.feature_names,
        "metrics": current_model.metrics,
        "feature_importance": current_model.get_feature_importance(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict delivery time for an order.

    This endpoint:
    1. Fetches features from the feature store
    2. Computes order-specific features
    3. Runs the model prediction
    4. Logs the request for monitoring
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature store not initialized")

    start_time = time.time()
    prediction_timestamp = request.timestamp or datetime.utcnow()

    try:
        # Get features
        features = feature_store.get_features_for_inference(
            customer_id=request.customer_id,
            store_id=request.store_id,
            restaurant_lat=request.latitude,
            restaurant_lon=request.longitude,
            delivery_lat=request.delivery_latitude,
            delivery_lon=request.delivery_longitude,
            placed_at=prediction_timestamp,
            order_total_cents=request.total,
            item_count=request.item_count or request.quantity,
            order_id=request.order_id,
            subtotal=request.subtotal,
            delivery_fee=request.delivery_fee,
            tip=request.tip,
            quantity=request.quantity,
            traffic_multiplier=request.traffic_multiplier,
            weather_condition=request.weather_condition,
            is_peak_hour=request.is_peak_hour,
        )

        # Predict with thread-safe model access
        # Brief lock ensures we get consistent model reference
        with model_lock:
            current_model = model
            model_version = current_model.version if current_model else "unknown"
        
        prediction = current_model.predict(features)

        # Log for monitoring
        latency_ms = (time.time() - start_time) * 1000
        log_entry = {
            "timestamp": prediction_timestamp.isoformat(),
            "order_id": request.order_id,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "features": features,
            "model_version": model_version,
        }
        request_log.append(log_entry)

        # Keep only last 1000 requests in memory
        if len(request_log) > 1000:
            request_log.pop(0)

        return PredictionResponse(
            order_id=request.order_id,
            predicted_delivery_minutes=prediction,
            prediction_timestamp=prediction_timestamp,
            model_version=model_version,
            features_used=features,
        )

    except Exception as e:
        # Log error
        request_log.append(
            {
                "timestamp": prediction_timestamp.isoformat(),
                "order_id": request.order_id,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/requests")
async def get_request_log(limit: int = 100) -> list[dict[str, Any]]:
    """Get recent prediction requests for monitoring."""
    return request_log[-limit:]


@app.get("/monitoring/stats")
async def get_stats() -> dict[str, Any]:
    """Get prediction statistics."""
    if not request_log:
        return {"total_requests": 0}

    successful = [r for r in request_log if "prediction" in r]
    errors = [r for r in request_log if "error" in r]

    if successful:
        latencies = [r["latency_ms"] for r in successful]
        predictions = [r["prediction"] for r in successful]

        return {
            "total_requests": len(request_log),
            "successful_requests": len(successful),
            "error_requests": len(errors),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "avg_prediction_minutes": sum(predictions) / len(predictions),
        }

    return {
        "total_requests": len(request_log),
        "successful_requests": 0,
        "error_requests": len(errors),
    }


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict delivery times for a batch of confirmed orders.
    
    This endpoint receives confirmed orders from your service and:
    1. Generates predictions for each order
    2. Saves predictions to local predictions.db
    3. Returns all predictions in the response
    
    Use this when orders are confirmed in your other service.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature store not initialized")
    
    # Get model reference with lock (brief lock doesn't block retraining)
    with model_lock:
        current_model = model
        model_version = current_model.version if current_model else "unknown"
    
    predictions = []
    errors = []
    successful = 0
    
    for order in request.orders:
        try:
            # Parse timestamp
            prediction_timestamp = datetime.fromisoformat(order.created_at)
            
            # Get features
            features = feature_store.get_features_for_inference(
                customer_id=order.customer_id,
                store_id=order.store_id,
                restaurant_lat=order.store_latitude,
                restaurant_lon=order.store_longitude,
                delivery_lat=order.delivery_latitude,
                delivery_lon=order.delivery_longitude,
                placed_at=prediction_timestamp,
                order_total_cents=order.total,
                item_count=order.item_count or order.quantity or 1,
                order_id=order.order_id,
                subtotal=order.subtotal or 0.0,
                delivery_fee=order.delivery_fee or 0.0,
                tip=order.tip or 0.0,
                quantity=order.quantity or 1,
                traffic_multiplier=order.traffic_multiplier or 1.0,
                weather_condition=order.weather_condition,
                is_peak_hour=order.is_peak_hour or False,
            )
            
            # Make prediction (no lock needed - using cached reference)
            prediction = current_model.predict(features)
            
            # Save to database
            _save_prediction_to_db(
                order_id=order.order_id,
                customer_id=order.customer_id,
                store_id=order.store_id,
                prediction=prediction,
                features=features,
                model_version=model_version,
                timestamp=prediction_timestamp,
            )
            
            # Add to response
            predictions.append(
                OrderPrediction(
                    order_id=order.order_id,
                    predicted_delivery_minutes=prediction,
                    model_version=model_version,
                    prediction_timestamp=prediction_timestamp.isoformat(),
                )
            )
            successful += 1
            
        except Exception as e:
            errors.append({
                "order_id": order.order_id,
                "error": str(e),
            })
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_orders=len(request.orders),
        successful=successful,
        failed=len(errors),
        errors=errors,
    )


@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 100) -> dict[str, Any]:
    """
    Get recent predictions from the database.
    
    Useful for monitoring what predictions have been made.
    """
    conn = sqlite3.connect(predictions_db_path)
    try:
        cursor = conn.execute(
            """
            SELECT 
                order_id,
                customer_id,
                store_id,
                predicted_delivery_minutes,
                prediction_timestamp,
                model_version,
                created_at
            FROM ml_predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "order_id": row[0],
                "customer_id": row[1],
                "store_id": row[2],
                "predicted_delivery_minutes": row[3],
                "prediction_timestamp": row[4],
                "model_version": row[5],
                "created_at": row[6],
            })
        
        return {
            "count": len(results),
            "predictions": results,
        }
    finally:
        conn.close()


@app.get("/predictions/stats")
async def get_predictions_stats() -> dict[str, Any]:
    """
    Get statistics about stored predictions.
    """
    conn = sqlite3.connect(predictions_db_path)
    try:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(DISTINCT order_id) as unique_orders,
                AVG(predicted_delivery_minutes) as avg_prediction,
                MIN(predicted_delivery_minutes) as min_prediction,
                MAX(predicted_delivery_minutes) as max_prediction,
                MIN(prediction_timestamp) as earliest_prediction,
                MAX(prediction_timestamp) as latest_prediction
            FROM ml_predictions
        """)
        
        row = cursor.fetchone()
        return {
            "total_predictions": row[0],
            "unique_orders": row[1],
            "avg_prediction_minutes": row[2],
            "min_prediction_minutes": row[3],
            "max_prediction_minutes": row[4],
            "earliest_prediction": row[5],
            "latest_prediction": row[6],
        }
    finally:
        conn.close()


def run_server() -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "delivery_ml.serving.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run_server()
