"""Feature store implementation.

Two-layer architecture:
1. Offline store (SQLite - your grocery_delivery.db): Historical feature values for training
2. Online store (Redis): Low-latency feature serving for inference

The offline store is the source of truth. The online store is populated
from the offline store and serves cached features for real-time inference.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import redis

from delivery_ml.config import settings
from delivery_ml.features.computation import FeatureComputer


class OfflineFeatureStore:
    """
    Offline feature store backed by SQLite (grocery_delivery.db).

    Responsibilities:
    - Read raw events from your existing database
    - Compute and store materialized features
    - Generate point-in-time correct training datasets
    
    NOTE: This assumes your grocery_delivery.db has an 'orders' table.
    Run discover_schema() to see actual table structure.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(settings.sqlite_db_path)
        self._conn: sqlite3.Connection | None = None
        self.computer = FeatureComputer(self.db_path)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def discover_schema(self) -> dict[str, Any]:
        """
        Discover the schema of your grocery_delivery.db.
        
        Use this to understand your table structure before running training.
        """
        return self.computer.discover_schema()

    def initialize_schema(self) -> None:
        """
        Create ML-specific tables if they don't exist.
        
        This does NOT modify your existing tables - it only adds
        new tables for feature materialization and metadata.
        """
        # Materialized feature tables for entity-level aggregates
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_restaurant_features (
                store_id TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                window_days INTEGER NOT NULL,
                avg_delivery_minutes REAL,
                order_count INTEGER,
                PRIMARY KEY (store_id, computed_at, window_days)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_customer_features (
                customer_id TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                window_days INTEGER NOT NULL,
                order_count INTEGER,
                avg_delivery_minutes REAL,
                PRIMARY KEY (customer_id, computed_at, window_days)
            )
        """)

        # Feature versioning table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_feature_metadata (
                feature_name TEXT PRIMARY KEY,
                version INTEGER NOT NULL DEFAULT 1,
                definition_hash TEXT,
                last_computed_at TEXT,
                row_count INTEGER
            )
        """)
        
        self.conn.commit()

    def check_orders_table(self) -> dict[str, Any]:
        """
        Check if orders table exists and has expected columns.
        
        Returns info about what columns exist vs what we expect.
        """
        expected_columns = {
            "order_id", "customer_id", "store_id", "created_at",
            "delivered_at", "latitude",
            "longitude", "delivery_latitude", "delivery_longitude",
            "total", "quantity"
        }
        
        cursor = self.conn.execute("PRAGMA table_info(orders)")
        actual_columns = {row[1] for row in cursor.fetchall()}
        
        return {
            "table_exists": len(actual_columns) > 0,
            "expected_columns": expected_columns,
            "actual_columns": actual_columns,
            "missing_columns": expected_columns - actual_columns,
            "extra_columns": actual_columns - expected_columns,
        }

    def materialize_restaurant_features(
        self, as_of: datetime, window_days: int = 30
    ) -> int:
        """
        Materialize restaurant features as of a specific time.

        Returns the number of restaurants processed.
        """
        as_of_str = as_of.strftime("%Y-%m-%d %H:%M:%S")
        window_start = as_of - timedelta(days=window_days)
        window_start_str = window_start.strftime("%Y-%m-%d %H:%M:%S")
        
        self.conn.execute(
            """
            INSERT OR REPLACE INTO ml_restaurant_features
            SELECT
                store_id,
                ? as computed_at,
                ? as window_days,
                AVG(delivery_time_minutes) as avg_delivery_minutes,
                COUNT(*) as order_count
            FROM orders
            WHERE delivered_at IS NOT NULL
              AND delivered_at >= ?
              AND delivered_at < ?
            GROUP BY store_id
            """,
            (as_of_str, window_days, window_start_str, as_of_str),
        )
        self.conn.commit()

        cursor = self.conn.execute(
            "SELECT COUNT(DISTINCT store_id) FROM ml_restaurant_features WHERE computed_at = ?",
            (as_of_str,),
        )
        result = cursor.fetchone()

        return result[0] if result else 0

    def get_training_data(
        self, start_date: datetime, end_date: datetime
    ) -> pl.DataFrame:
        """Get training data with point-in-time correct features."""
        return self.computer.compute_training_features_batch_optimized(
            start_date, end_date
        )

    def get_order_count(self) -> int:
        """Get total number of orders in the database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM orders")
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_date_range(self) -> tuple[datetime | None, datetime | None]:
        """Get the date range of orders in the database."""
        cursor = self.conn.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM orders"
        )
        result = cursor.fetchone()
        
        if result and result[0] and result[1]:
            min_date = datetime.fromisoformat(result[0]) if isinstance(result[0], str) else result[0]
            max_date = datetime.fromisoformat(result[1]) if isinstance(result[1], str) else result[1]
            return min_date, max_date
        
        return None, None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self.computer.close()


class OnlineFeatureStore:
    """
    Online feature store backed by Redis.

    Optimized for low-latency feature retrieval at inference time.
    Features are keyed by entity_type:entity_id:feature_name.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ):
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True,
            )
        return self._client

    def is_available(self) -> bool:
        """Check if Redis is available."""
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    def _make_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        """Create a Redis key for a feature."""
        return f"feature:{entity_type}:{entity_id}:{feature_name}"

    def set_feature(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set a single feature value."""
        key = self._make_key(entity_type, entity_id, feature_name)
        serialized = json.dumps({"value": value, "updated_at": datetime.utcnow().isoformat()})

        if ttl_seconds:
            self.client.setex(key, ttl_seconds, serialized)
        else:
            self.client.set(key, serialized)

    def get_feature(
        self, entity_type: str, entity_id: str, feature_name: str
    ) -> Any | None:
        """Get a single feature value."""
        key = self._make_key(entity_type, entity_id, feature_name)
        data = self.client.get(key)

        if data is None:
            return None

        return json.loads(data)["value"]

    def set_features_bulk(
        self,
        entity_type: str,
        entity_id: str,
        features: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Set multiple features for an entity."""
        pipe = self.client.pipeline()
        updated_at = datetime.utcnow().isoformat()

        for feature_name, value in features.items():
            key = self._make_key(entity_type, entity_id, feature_name)
            serialized = json.dumps({"value": value, "updated_at": updated_at})

            if ttl_seconds:
                pipe.setex(key, ttl_seconds, serialized)
            else:
                pipe.set(key, serialized)

        pipe.execute()

    def get_features_bulk(
        self, entity_type: str, entity_id: str, feature_names: list[str]
    ) -> dict[str, Any | None]:
        """Get multiple features for an entity."""
        keys = [self._make_key(entity_type, entity_id, name) for name in feature_names]
        values = self.client.mget(keys)

        result = {}
        for name, value in zip(feature_names, values):
            if value is None:
                result[name] = None
            else:
                result[name] = json.loads(value)["value"]

        return result

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class FeatureStore:
    """
    Unified feature store interface.

    Coordinates between offline and online stores.
    """

    def __init__(self, db_path: str | None = None):
        self.offline = OfflineFeatureStore(db_path)
        self.online = OnlineFeatureStore()
        self.computer = FeatureComputer(db_path)
        self._redis_available: bool | None = None

    def initialize(self) -> None:
        """Initialize the feature store."""
        self.offline.initialize_schema()
        
        # Check Redis availability once
        self._redis_available = self.online.is_available()
        if not self._redis_available:
            print("Warning: Redis not available. Online features will fall back to SQLite.")

    def sync_to_online(self, entity_type: str, entity_id: str, as_of: datetime) -> None:
        """
        Sync features for an entity from offline to online store.

        This is called periodically (e.g., daily) to refresh the online store.
        """
        if not self._redis_available:
            return  # Skip if Redis not available
            
        if entity_type == "restaurant":
            features = self.computer.compute_restaurant_features_at_time(
                entity_id, as_of, window_days=30
            )
        elif entity_type == "customer":
            features = self.computer.compute_customer_features_at_time(
                entity_id, as_of, window_days=30
            )
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

        self.online.set_features_bulk(
            entity_type, entity_id, features, ttl_seconds=86400  # 24 hours
        )

    def get_features_for_inference(
        self,
        customer_id: str,
        store_id: str,
        restaurant_lat: float,
        restaurant_lon: float,
        delivery_lat: float,
        delivery_lon: float,
        placed_at: datetime,
        order_total_cents: int,
        item_count: int,
    ) -> dict[str, Any]:
        """
        Get all features needed for inference.

        Combines:
        - Cached entity features from online store (if Redis available)
        - Real-time computed order features
        """
        restaurant_features = None
        customer_features = None
        
        # Try to get cached features from Redis if available
        if self._redis_available:
            try:
                restaurant_features = self.online.get_features_bulk(
                    "restaurant",
                    store_id,
                    ["restaurant_avg_delivery_minutes_30d", "restaurant_order_count_30d"],
                )
                
                # Check if we got valid cached data
                if restaurant_features.get("restaurant_avg_delivery_minutes_30d") is None:
                    restaurant_features = None
                    
                customer_features = self.online.get_features_bulk(
                    "customer", customer_id, ["customer_order_count_30d"]
                )
                
                if customer_features.get("customer_order_count_30d") is None:
                    customer_features = None
            except Exception:
                pass  # Fall back to SQLite

        # Fall back to computing from SQLite if not cached
        if restaurant_features is None:
            restaurant_features = self.computer.compute_restaurant_features_at_time(
                store_id, placed_at, window_days=30
            )

        if customer_features is None:
            customer_features = self.computer.compute_customer_features_at_time(
                customer_id, placed_at, window_days=30
            )

        # Always compute order features (they're specific to this order)
        order_features = self.computer.compute_order_features(
            restaurant_lat,
            restaurant_lon,
            delivery_lat,
            delivery_lon,
            placed_at,
            order_total_cents,
            item_count,
        )

        return {**restaurant_features, **customer_features, **order_features}

    def close(self) -> None:
        self.offline.close()
        if self._redis_available:
            self.online.close()
        self.computer.close()
