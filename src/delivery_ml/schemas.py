"""Data schemas and validation for delivery events and features."""

from datetime import datetime
from enum import Enum

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Event Schemas (Pydantic) - for API inputs and event ingestion
# -----------------------------------------------------------------------------


class OrderPlaced(BaseModel):
    """Event: customer places an order."""

    order_id: str
    customer_id: str
    store_id: str
    driver_id: str | None = None
    timestamp: datetime
    delivery_latitude: float
    delivery_longitude: float
    latitude: float
    longitude: float
    total: int = Field(ge=0)
    quantity: int = Field(ge=1)


class OrderAssigned(BaseModel):
    """Event: order is assigned to a driver."""

    order_id: str
    driver_id: str
    timestamp: datetime


class OrderPickedUp(BaseModel):
    """Event: driver picks up the order from restaurant."""

    order_id: str
    timestamp: datetime


class OrderDelivered(BaseModel):
    """Event: order is delivered to customer."""

    order_id: str
    timestamp: datetime


class OrderCancelled(BaseModel):
    """Event: order is cancelled."""

    order_id: str
    timestamp: datetime
    reason: str | None = None


# -----------------------------------------------------------------------------
# Feature Schemas (Pandera) - for DataFrame validation
# -----------------------------------------------------------------------------


class OrderEventsSchema(pa.DataFrameModel):
    """Schema for the orders event table."""

    order_id: str = pa.Field(unique=True)
    customer_id: str
    store_id: str
    driver_id: str = pa.Field(nullable=True)
    placed_at: pl.Datetime
    assigned_at: pl.Datetime = pa.Field(nullable=True)
    picked_up_at: pl.Datetime = pa.Field(nullable=True)
    delivered_at: pl.Datetime = pa.Field(nullable=True)
    cancelled_at: pl.Datetime = pa.Field(nullable=True)
    delivery_latitude: float
    delivery_longitude: float
    latitude: float
    longitude: float
    total: int = pa.Field(ge=0)
    quantity: int = pa.Field(ge=1)


class TrainingFeaturesSchema(pa.DataFrameModel):
    """Schema for training feature set."""

    order_id: str = pa.Field(unique=True)

    # Target
    delivery_time_minutes: float = pa.Field(ge=0, nullable=True)

    # Features
    distance_km: float = pa.Field(ge=0)
    traffic_multiplier: float = pa.Field(ge=0, nullable=True)
    weather_condition: int = pa.Field(ge=0, nullable=True)
    speed_multiplier: float = pa.Field(nullable=True)
    hour_of_day: int = pa.Field(ge=0, le=23)
    is_weekend: bool
    is_peak_hour: bool
    experience_level: int = pa.Field(ge=0, nullable=True)
    total_deliveries: int = pa.Field(ge=0, nullable=True)


# -----------------------------------------------------------------------------
# Inference Schemas
# -----------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Request for delivery time prediction."""

    order_id: str
    customer_id: str
    store_id: str
    delivery_latitude: float
    delivery_longitude: float
    latitude: float
    longitude: float
    total: int = Field(ge=0)
    quantity: int = Field(ge=1)
    subtotal: float = Field(default=0.0, ge=0)
    delivery_fee: float = Field(default=0.0, ge=0)
    tip: float = Field(default=0.0, ge=0)
    item_count: int = Field(default=0, ge=0)
    traffic_multiplier: float = Field(default=1.0, ge=0)
    weather_condition: str | None = Field(default=None)
    is_peak_hour: bool = Field(default=False)
    timestamp: datetime | None = None  # defaults to now


class PredictionResponse(BaseModel):
    """Response with predicted delivery time."""

    order_id: str
    predicted_delivery_minutes: float
    prediction_timestamp: datetime
    model_version: str
    features_used: dict[str, float | int | bool]
