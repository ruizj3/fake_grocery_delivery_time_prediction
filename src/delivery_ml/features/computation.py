"""Feature computation logic.

This module implements HOW features are computed from raw events.
Key concepts:
- Point-in-time correctness: features only use data available at prediction time
- Batch computation: for training data generation
- Online computation: for real-time serving

Updated to work with SQLite (grocery_delivery.db).
"""

import math
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from delivery_ml.config import settings


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the haversine distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Encoding maps for categorical features
WEATHER_ENCODING: dict[str | None, int] = {
    None: 0,
    "clear": 0,
    "cloudy": 1,
    "rain": 2,
    "snow": 3,
    "storm": 4,
}

EXPERIENCE_ENCODING: dict[str | None, int] = {
    None: 0,
    "intermediate": 0,
    "advanced": 1,
    "expert": 2,
}


def encode_weather(value: str | None) -> int:
    """Encode weather condition string to integer."""
    if value is None:
        return WEATHER_ENCODING[None]
    return WEATHER_ENCODING.get(value.lower().strip(), 0)


def encode_experience(value: str | None) -> int:
    """Encode driver experience level string to integer."""
    if value is None:
        return EXPERIENCE_ENCODING[None]
    return EXPERIENCE_ENCODING.get(value.lower().strip(), 0)


class FeatureComputer:
    """Computes features from raw events stored in SQLite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(settings.sqlite_db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _get_table_info(self) -> dict[str, list[str]]:
        """Get information about tables in the database."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {}
        for row in cursor.fetchall():
            table_name = row[0]
            col_cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
            tables[table_name] = [col[1] for col in col_cursor.fetchall()]
        return tables

    def discover_schema(self) -> dict[str, Any]:
        """
        Discover the schema of your grocery_delivery.db.
        
        Call this to understand what tables and columns exist so we can
        map them to features.
        """
        tables = self._get_table_info()
        result = {"tables": {}}
        
        for table_name, columns in tables.items():
            # Get sample data
            cursor = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample = [dict(row) for row in cursor.fetchall()]
            
            # Get row count
            count_cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = count_cursor.fetchone()[0]
            
            result["tables"][table_name] = {
                "columns": columns,
                "row_count": count,
                "sample": sample,
            }
        
        return result

    # -------------------------------------------------------------------------
    # Point-in-Time Feature Computation
    # These methods assume your database has an 'orders' table with certain columns.
    # You may need to adjust column names to match your actual schema.
    # -------------------------------------------------------------------------

    def compute_restaurant_features_at_time(
        self,
        store_id: str,
        as_of: datetime,
        window_days: int = 30,
    ) -> dict[str, float | int | None]:
        """
        Compute restaurant features as of a specific point in time.

        This is the critical point-in-time correctness piece:
        we only use orders that were COMPLETED before as_of.
        
        NOTE: delivery_time_minutes is calculated from (delivered_at - created_at)
        """
        window_start = as_of - timedelta(days=window_days)
        
        # Convert to string format SQLite understands
        window_start_str = window_start.strftime("%Y-%m-%d %H:%M:%S")
        as_of_str = as_of.strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.execute(
            """
            SELECT
                AVG((julianday(delivered_at) - julianday(created_at)) * 24 * 60) as avg_delivery_minutes,
                COUNT(*) as order_count
            FROM orders
            WHERE store_id = ?
              AND delivered_at IS NOT NULL
              AND delivered_at >= ?
              AND delivered_at < ?
            """,
            (store_id, window_start_str, as_of_str),
        )
        
        result = cursor.fetchone()

        if result is None or result["order_count"] == 0:
            return {
                f"restaurant_avg_delivery_minutes_{window_days}d": 0.0,
                f"restaurant_order_count_{window_days}d": 0,
            }

        return {
            f"restaurant_avg_delivery_minutes_{window_days}d": float(result["avg_delivery_minutes"] or 0.0),
            f"restaurant_order_count_{window_days}d": result["order_count"] or 0,
        }

    def compute_customer_features_at_time(
        self,
        customer_id: str,
        as_of: datetime,
        window_days: int = 30,
    ) -> dict[str, float | int | None]:
        """Compute customer features as of a specific point in time."""
        window_start = as_of - timedelta(days=window_days)
        
        window_start_str = window_start.strftime("%Y-%m-%d %H:%M:%S")
        as_of_str = as_of.strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.execute(
            """
            SELECT
                AVG((julianday(delivered_at) - julianday(created_at)) * 24 * 60) as avg_delivery_minutes,
                COUNT(*) as order_count
            FROM orders
            WHERE customer_id = ?
              AND delivered_at IS NOT NULL
              AND delivered_at >= ?
              AND delivered_at < ?
            """,
            (customer_id, window_start_str, as_of_str),
        )
        
        result = cursor.fetchone()

        if result is None or result["order_count"] == 0:
            return {
                f"customer_order_count_{window_days}d": 0,
            }

        return {
            f"customer_order_count_{window_days}d": result["order_count"] or 0,
        }

    def compute_order_features(
        self,
        restaurant_lat: float,
        restaurant_lon: float,
        delivery_lat: float,
        delivery_lon: float,
        placed_at: datetime,
        order_total_cents: int,
        quantity: int,
        subtotal: float = 0.0,
        delivery_fee: float = 0.0,
        tip: float = 0.0,
        item_count: int = 0,
        traffic_multiplier: float = 1.0,
        weather_condition: str | None = None,
        is_peak_hour: bool = False,
    ) -> dict[str, float | int | bool]:
        """Compute features derived directly from order attributes."""
        return {
            "distance_km": haversine_distance_km(
                restaurant_lat, restaurant_lon, delivery_lat, delivery_lon
            ),
            "hour_of_day": placed_at.hour,
            "day_of_week": placed_at.weekday(),
            "is_weekend": placed_at.weekday() >= 5,
            "total": order_total_cents,
            "subtotal": subtotal,
            "delivery_fee": delivery_fee,
            "tip": tip,
            "quantity": quantity,
            "item_count": item_count or quantity,
            "traffic_multiplier": traffic_multiplier,
            "weather_condition": encode_weather(weather_condition),
            "is_peak_hour": is_peak_hour,
        }

    def compute_driver_features_for_order(
        self,
        order_id: str,
    ) -> dict[str, float | int | None]:
        """
        Compute driver-related features for an order.

        Looks up the driver assigned via bundles and gets their
        reliability_score, speed_multiplier, experience_level, and total_deliveries.
        """
        cursor = self.conn.execute(
            """
            SELECT d.reliability_score, d.speed_multiplier,
                   d.experience_level, d.total_deliveries
            FROM bundles b
            JOIN bundle_stops bs ON b.bundle_id = bs.bundle_id
            JOIN drivers d ON b.driver_id = d.driver_id
            WHERE bs.order_id = ?
            LIMIT 1
            """,
            (order_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "reliability_score": float(row["reliability_score"]) if row["reliability_score"] is not None else 0.0,
                "speed_multiplier": float(row["speed_multiplier"]) if row["speed_multiplier"] is not None else 1.0,
                "experience_level": encode_experience(row["experience_level"]),
                "total_deliveries": int(row["total_deliveries"]) if row["total_deliveries"] is not None else 0,
            }
        return {"reliability_score": 0.0, "speed_multiplier": 1.0, "experience_level": 0, "total_deliveries": 0}

    def compute_bundle_features_for_order(
        self,
        order_id: str,
    ) -> dict[str, int | None]:
        """
        Compute bundle/route features for an order.

        Gets stops_in_bundle (total stops) and stop_sequence (this order's position).
        """
        cursor = self.conn.execute(
            """
            SELECT
                bs.stop_sequence,
                b.order_count as stops_in_bundle
            FROM bundle_stops bs
            JOIN bundles b ON bs.bundle_id = b.bundle_id
            WHERE bs.order_id = ?
            LIMIT 1
            """,
            (order_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "stop_sequence": int(row["stop_sequence"]) if row["stop_sequence"] is not None else 1,
                "stops_in_bundle": int(row["stops_in_bundle"]) if row["stops_in_bundle"] is not None else 1,
            }
        return {"stop_sequence": 1, "stops_in_bundle": 1}

    def compute_features_for_order(
        self,
        order_id: str,
        customer_id: str,
        store_id: str,
        restaurant_lat: float,
        restaurant_lon: float,
        delivery_lat: float,
        delivery_lon: float,
        placed_at: datetime,
        order_total_cents: int,
        quantity: int,
        subtotal: float = 0.0,
        delivery_fee: float = 0.0,
        tip: float = 0.0,
        item_count: int = 0,
        traffic_multiplier: float = 1.0,
        weather_condition: str | None = None,
        is_peak_hour: bool = False,
    ) -> dict[str, float | int | bool | None]:
        """
        Compute all features for a single order at prediction time.

        Used for online serving.
        """
        features: dict[str, float | int | bool | None] = {"order_id": order_id}

        # Restaurant features (point-in-time)
        features.update(
            self.compute_restaurant_features_at_time(store_id, placed_at, window_days=30)
        )

        # Customer features (point-in-time)
        features.update(
            self.compute_customer_features_at_time(customer_id, placed_at, window_days=30)
        )

        # Order features (computed from order attributes)
        features.update(
            self.compute_order_features(
                restaurant_lat,
                restaurant_lon,
                delivery_lat,
                delivery_lon,
                placed_at,
                order_total_cents,
                quantity,
                subtotal=subtotal,
                delivery_fee=delivery_fee,
                tip=tip,
                item_count=item_count,
                traffic_multiplier=traffic_multiplier,
                weather_condition=weather_condition,
                is_peak_hour=is_peak_hour,
            )
        )

        # Driver features (from bundles/drivers tables)
        features.update(self.compute_driver_features_for_order(order_id))

        # Bundle/route features
        features.update(self.compute_bundle_features_for_order(order_id))

        return features

    # -------------------------------------------------------------------------
    # Batch Feature Computation (for training)
    # -------------------------------------------------------------------------

    def compute_training_features_batch(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """
        Compute features for all orders in a date range.

        This is the batch computation used for training data generation.
        For each order, we compute features using only data available
        at the time the order was placed (point-in-time correctness).
        """
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get all completed orders in the date range
        # Calculate delivery_time_minutes from timestamps
        cursor = self.conn.execute(
            """
            SELECT
                o.order_id,
                o.customer_id,
                o.store_id,
                o.created_at,
                o.delivered_at,
                (julianday(o.delivered_at) - julianday(o.created_at)) * 24 * 60 as delivery_time_minutes,
                s.latitude,
                s.longitude,
                o.delivery_latitude,
                o.delivery_longitude,
                o.total,
                COALESCE(o.subtotal, 0) as subtotal,
                COALESCE(o.delivery_fee, 0) as delivery_fee,
                COALESCE(o.tip, 0) as tip,
                COALESCE(oi.quantity, 1) as quantity,
                COALESCE(oi.item_count, 1) as item_count
            FROM orders o
            LEFT JOIN stores s ON o.store_id = s.store_id
            LEFT JOIN (
                SELECT order_id, SUM(quantity) AS quantity, COUNT(*) AS item_count
                FROM order_items GROUP BY order_id
            ) oi ON o.order_id = oi.order_id
            WHERE o.delivered_at IS NOT NULL
              AND o.created_at >= ?
              AND o.created_at < ?
            ORDER BY o.created_at
            """,
            (start_str, end_str),
        )
        
        rows = cursor.fetchall()

        if not rows:
            return pl.DataFrame()

        # Compute features for each order
        feature_rows = []

        for row in rows:
            row_dict = dict(row)
            
            # Parse datetime strings if needed
            placed_at = row_dict["created_at"]
            if isinstance(placed_at, str):
                placed_at = datetime.fromisoformat(placed_at)
            
            features = self.compute_features_for_order(
                order_id=row_dict["order_id"],
                customer_id=row_dict["customer_id"],
                store_id=row_dict["store_id"],
                restaurant_lat=row_dict["latitude"],
                restaurant_lon=row_dict["longitude"],
                delivery_lat=row_dict["delivery_latitude"],
                delivery_lon=row_dict["delivery_longitude"],
                placed_at=placed_at,
                order_total_cents=row_dict["total"],
                quantity=row_dict["quantity"],
                subtotal=row_dict["subtotal"],
                delivery_fee=row_dict["delivery_fee"],
                tip=row_dict["tip"],
                item_count=row_dict["item_count"],
            )
            features["delivery_time_minutes"] = row_dict["delivery_time_minutes"]
            feature_rows.append(features)

        return pl.DataFrame(feature_rows)

    def compute_training_features_batch_optimized(
        self,
        start_date: datetime,
        end_date: datetime,
        window_days: int = 30,
    ) -> pl.DataFrame:
        """
        Optimized batch feature computation using SQL.

        SQLite doesn't have as rich window function support as DuckDB,
        so we use a hybrid approach: SQL for base data, Python for aggregates.
        
        For very large datasets, consider switching to DuckDB or doing
        the aggregation in chunks.
        """
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get base order data with order features computed in SQL
        # Includes new features: subtotal, delivery_fee, tip, item_count,
        # driver reliability/speed, and bundle stop info
        cursor = self.conn.execute(
            """
            SELECT
                o.order_id,
                o.customer_id,
                o.store_id,
                o.created_at,
                o.delivered_at,
                (julianday(o.delivered_at) - julianday(o.created_at)) * 24 * 60 as delivery_time_minutes,
                s.latitude,
                s.longitude,
                o.delivery_latitude,
                o.delivery_longitude,
                o.total,
                COALESCE(o.subtotal, 0) as subtotal,
                COALESCE(o.delivery_fee, 0) as delivery_fee,
                COALESCE(o.tip, 0) as tip,
                COALESCE(oi.quantity, 1) as quantity,
                COALESCE(oi.item_count, 1) as item_count,
                CAST(strftime('%H', o.created_at) AS INTEGER) as hour_of_day,
                CAST(strftime('%w', o.created_at) AS INTEGER) as day_of_week,
                COALESCE(d.reliability_score, 0) as reliability_score,
                COALESCE(d.speed_multiplier, 1) as speed_multiplier,
                COALESCE(b.order_count, 1) as stops_in_bundle,
                COALESCE(bs.stop_sequence, 1) as stop_sequence,
                COALESCE(o.traffic_multiplier, 1.0) as traffic_multiplier,
                o.weather_condition,
                COALESCE(o.is_peak_hour, 0) as is_peak_hour,
                COALESCE(o.is_weekend, 0) as is_weekend_db,
                d.experience_level,
                COALESCE(d.total_deliveries, 0) as total_deliveries
            FROM orders o
            LEFT JOIN stores s ON o.store_id = s.store_id
            LEFT JOIN (
                SELECT order_id, SUM(quantity) AS quantity, COUNT(*) AS item_count
                FROM order_items GROUP BY order_id
            ) oi ON o.order_id = oi.order_id
            LEFT JOIN bundle_stops bs ON o.order_id = bs.order_id
            LEFT JOIN bundles b ON bs.bundle_id = b.bundle_id
            LEFT JOIN drivers d ON b.driver_id = d.driver_id
            WHERE o.delivered_at IS NOT NULL
              AND o.created_at >= ?
              AND o.created_at < ?
            ORDER BY o.created_at
            """,
            (start_str, end_str),
        )
        
        rows = cursor.fetchall()
        
        if not rows:
            return pl.DataFrame()

        # Build feature rows with aggregations
        feature_rows = []
        
        for row in rows:
            row_dict = dict(row)
            
            placed_at = row_dict["created_at"]
            if isinstance(placed_at, str):
                placed_at = datetime.fromisoformat(placed_at)
            
            # Get restaurant aggregates (point-in-time)
            restaurant_features = self.compute_restaurant_features_at_time(
                row_dict["store_id"], placed_at, window_days
            )
            
            # Get customer aggregates (point-in-time)
            customer_features = self.compute_customer_features_at_time(
                row_dict["customer_id"], placed_at, window_days
            )
            
            # Compute distance
            distance_km = haversine_distance_km(
                row_dict["latitude"],
                row_dict["longitude"],
                row_dict["delivery_latitude"],
                row_dict["delivery_longitude"],
            )
            
            # SQLite strftime('%w') returns 0=Sunday, we want 0=Monday
            sqlite_dow = row_dict["day_of_week"]
            day_of_week = (sqlite_dow - 1) % 7 if sqlite_dow > 0 else 6
            
            feature_row = {
                "order_id": row_dict["order_id"],
                "delivery_time_minutes": row_dict["delivery_time_minutes"],
                "distance_km": distance_km,
                "hour_of_day": row_dict["hour_of_day"],
                "day_of_week": day_of_week,
                "is_weekend": bool(row_dict["is_weekend_db"]) if row_dict["is_weekend_db"] else day_of_week >= 5,
                "total": row_dict["total"],
                "subtotal": row_dict["subtotal"],
                "delivery_fee": row_dict["delivery_fee"],
                "tip": row_dict["tip"],
                "quantity": row_dict["quantity"],
                "item_count": row_dict["item_count"],
                "reliability_score": row_dict["reliability_score"],
                "speed_multiplier": row_dict["speed_multiplier"],
                "stops_in_bundle": row_dict["stops_in_bundle"],
                "stop_sequence": row_dict["stop_sequence"],
                "traffic_multiplier": row_dict["traffic_multiplier"],
                "weather_condition": encode_weather(row_dict["weather_condition"]),
                "is_peak_hour": bool(row_dict["is_peak_hour"]),
                "experience_level": encode_experience(row_dict["experience_level"]),
                "total_deliveries": row_dict["total_deliveries"],
                **restaurant_features,
                **customer_features,
            }
            
            feature_rows.append(feature_row)

        return pl.DataFrame(feature_rows)
