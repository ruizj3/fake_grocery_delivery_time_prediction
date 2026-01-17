"""FastAPI serving endpoint for delivery time predictions."""

import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from delivery_ml.config import settings
from delivery_ml.features.store import FeatureStore
from delivery_ml.schemas import PredictionRequest, PredictionResponse
from delivery_ml.training.pipeline import DeliveryTimeModel


# Global state
model: DeliveryTimeModel | None = None
feature_store: FeatureStore | None = None
predictions_db_path: Path = Path("predictions.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize feature store on startup."""
    global model, feature_store, predictions_db_path

    # Load model
    model_path = settings.model_dir / "delivery_time_model_latest.pkl"
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        model = None
    else:
        model = DeliveryTimeModel.load(model_path)
        print(f"Loaded model version: {model.version}")

    # Initialize feature store
    feature_store = FeatureStore()
    feature_store.initialize()
    print("Feature store initialized")
    
    # Initialize predictions database
    _initialize_predictions_db()
    print(f"Predictions database initialized at {predictions_db_path}")

    yield

    # Cleanup
    if feature_store:
        feature_store.close()


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
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_version": model.version if model else "none",
    }


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "version": model.version,
        "features": model.feature_names,
        "metrics": model.metrics,
        "feature_importance": model.get_feature_importance(),
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
            item_count=request.quantity,
        )

        # Predict
        prediction = model.predict(features)

        # Log for monitoring
        latency_ms = (time.time() - start_time) * 1000
        log_entry = {
            "timestamp": prediction_timestamp.isoformat(),
            "order_id": request.order_id,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "features": features,
            "model_version": model.version,
        }
        request_log.append(log_entry)

        # Keep only last 1000 requests in memory
        if len(request_log) > 1000:
            request_log.pop(0)

        return PredictionResponse(
            order_id=request.order_id,
            predicted_delivery_minutes=prediction,
            prediction_timestamp=prediction_timestamp,
            model_version=model.version,
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
                item_count=order.quantity or 1,
            )
            
            # Make prediction
            prediction = model.predict(features)
            
            # Save to database
            _save_prediction_to_db(
                order_id=order.order_id,
                customer_id=order.customer_id,
                store_id=order.store_id,
                prediction=prediction,
                features=features,
                model_version=model.version,
                timestamp=prediction_timestamp,
            )
            
            # Add to response
            predictions.append(
                OrderPrediction(
                    order_id=order.order_id,
                    predicted_delivery_minutes=prediction,
                    model_version=model.version,
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
