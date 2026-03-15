"""Feature definitions for delivery time prediction.

This module defines WHAT features we compute, not HOW we compute them.
Each feature definition includes:
- name: unique identifier
- description: what this feature represents
- entity: what entity this feature belongs to (restaurant, customer, driver, order)
- aggregation: how to compute this feature from raw events
- window: time window for aggregation (if applicable)
- freshness: how often this feature should be updated
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Literal


class Entity(str, Enum):
    """Entities that features can belong to."""

    RESTAURANT = "restaurant"
    CUSTOMER = "customer"
    DRIVER = "driver"
    ORDER = "order"  # point-in-time, no aggregation


class AggregationType(str, Enum):
    """Types of aggregations for features."""

    MEAN = "mean"
    COUNT = "count"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    STDDEV = "stddev"
    NONE = "none"  # for non-aggregated features


class FreshnessRequirement(str, Enum):
    """How fresh this feature needs to be."""

    REALTIME = "realtime"  # must be computed at request time
    HOURLY = "hourly"  # can be up to 1 hour stale
    DAILY = "daily"  # can be up to 1 day stale
    STATIC = "static"  # doesn't change (or changes very rarely)


@dataclass(frozen=True)
class FeatureDefinition:
    """Definition of a feature."""

    name: str
    description: str
    entity: Entity
    aggregation: AggregationType
    window_days: int | None  # None for non-windowed features
    freshness: FreshnessRequirement
    dtype: Literal["float", "int", "bool"]


# -----------------------------------------------------------------------------
# Feature Registry
# -----------------------------------------------------------------------------

FEATURE_DEFINITIONS: dict[str, FeatureDefinition] = {}


def register_feature(feature: FeatureDefinition) -> FeatureDefinition:
    """Register a feature definition."""
    FEATURE_DEFINITIONS[feature.name] = feature
    return feature


# -----------------------------------------------------------------------------
# Restaurant Features
# -----------------------------------------------------------------------------

restaurant_avg_delivery_minutes_30d = register_feature(
    FeatureDefinition(
        name="restaurant_avg_delivery_minutes_30d",
        description="Average delivery time for orders from this restaurant over the last 30 days",
        entity=Entity.RESTAURANT,
        aggregation=AggregationType.MEAN,
        window_days=30,
        freshness=FreshnessRequirement.DAILY,
        dtype="float",
    )
)

restaurant_order_count_30d = register_feature(
    FeatureDefinition(
        name="restaurant_order_count_30d",
        description="Number of orders from this restaurant in the last 30 days",
        entity=Entity.RESTAURANT,
        aggregation=AggregationType.COUNT,
        window_days=30,
        freshness=FreshnessRequirement.DAILY,
        dtype="int",
    )
)

restaurant_avg_delivery_minutes_7d = register_feature(
    FeatureDefinition(
        name="restaurant_avg_delivery_minutes_7d",
        description="Average delivery time for orders from this restaurant over the last 7 days",
        entity=Entity.RESTAURANT,
        aggregation=AggregationType.MEAN,
        window_days=7,
        freshness=FreshnessRequirement.DAILY,
        dtype="float",
    )
)


# -----------------------------------------------------------------------------
# Customer Features
# -----------------------------------------------------------------------------

customer_order_count_30d = register_feature(
    FeatureDefinition(
        name="customer_order_count_30d",
        description="Number of orders by this customer in the last 30 days",
        entity=Entity.CUSTOMER,
        aggregation=AggregationType.COUNT,
        window_days=30,
        freshness=FreshnessRequirement.DAILY,
        dtype="int",
    )
)

customer_avg_delivery_minutes_30d = register_feature(
    FeatureDefinition(
        name="customer_avg_delivery_minutes_30d",
        description="Average delivery time for this customer's orders over the last 30 days",
        entity=Entity.CUSTOMER,
        aggregation=AggregationType.MEAN,
        window_days=30,
        freshness=FreshnessRequirement.DAILY,
        dtype="float",
    )
)


# -----------------------------------------------------------------------------
# Order Features (point-in-time, no aggregation)
# -----------------------------------------------------------------------------

distance_km = register_feature(
    FeatureDefinition(
        name="distance_km",
        description="Haversine distance from restaurant to delivery location in kilometers",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="float",
    )
)

hour_of_day = register_feature(
    FeatureDefinition(
        name="hour_of_day",
        description="Hour of day when order was placed (0-23)",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="int",
    )
)

day_of_week = register_feature(
    FeatureDefinition(
        name="day_of_week",
        description="Day of week when order was placed (0=Monday, 6=Sunday)",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="int",
    )
)

is_weekend = register_feature(
    FeatureDefinition(
        name="is_weekend",
        description="Whether the order was placed on a weekend",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="bool",
    )
)

traffic_multiplier = register_feature(
    FeatureDefinition(
        name="traffic_multiplier",
        description="Traffic multiplier affecting delivery time (1.0 = normal)",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.REALTIME,
        dtype="float",
    )
)

weather_condition = register_feature(
    FeatureDefinition(
        name="weather_condition",
        description="Encoded weather condition at order time (0=clear, 1=cloudy, 2=rain, 3=snow, 4=storm)",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.REALTIME,
        dtype="int",
    )
)

is_peak_hour = register_feature(
    FeatureDefinition(
        name="is_peak_hour",
        description="Whether the order was placed during peak hours",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="bool",
    )
)


# -----------------------------------------------------------------------------
# Order Financial Features
# -----------------------------------------------------------------------------

subtotal = register_feature(
    FeatureDefinition(
        name="subtotal",
        description="Order subtotal before taxes and fees",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="float",
    )
)

delivery_fee = register_feature(
    FeatureDefinition(
        name="delivery_fee",
        description="Delivery fee charged for the order",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="float",
    )
)

tip = register_feature(
    FeatureDefinition(
        name="tip",
        description="Tip amount for the order",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="float",
    )
)

item_count = register_feature(
    FeatureDefinition(
        name="item_count",
        description="Number of unique line items in the order",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="int",
    )
)


# -----------------------------------------------------------------------------
# Driver Features
# -----------------------------------------------------------------------------

reliability_score = register_feature(
    FeatureDefinition(
        name="reliability_score",
        description="Driver reliability score (higher is better)",
        entity=Entity.DRIVER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.DAILY,
        dtype="float",
    )
)

speed_multiplier = register_feature(
    FeatureDefinition(
        name="speed_multiplier",
        description="Driver speed multiplier affecting delivery time",
        entity=Entity.DRIVER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.DAILY,
        dtype="float",
    )
)

experience_level = register_feature(
    FeatureDefinition(
        name="experience_level",
        description="Encoded driver experience level (0=intermediate, 1=advanced, 2=expert)",
        entity=Entity.DRIVER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.DAILY,
        dtype="int",
    )
)

total_deliveries = register_feature(
    FeatureDefinition(
        name="total_deliveries",
        description="Total number of deliveries completed by the driver",
        entity=Entity.DRIVER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.DAILY,
        dtype="int",
    )
)


# -----------------------------------------------------------------------------
# Bundle/Route Features
# -----------------------------------------------------------------------------

stops_in_bundle = register_feature(
    FeatureDefinition(
        name="stops_in_bundle",
        description="Number of stops in the delivery bundle for this order",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="int",
    )
)

stop_sequence = register_feature(
    FeatureDefinition(
        name="stop_sequence",
        description="Position of this order in the delivery bundle route",
        entity=Entity.ORDER,
        aggregation=AggregationType.NONE,
        window_days=None,
        freshness=FreshnessRequirement.STATIC,
        dtype="int",
    )
)


# -----------------------------------------------------------------------------
# Feature Sets (groups of features used together)
# -----------------------------------------------------------------------------

TRAINING_FEATURES = [
    "distance_km",           # Haversine distance from store to delivery location
    "traffic_multiplier",    # Traffic condition multiplier (from orders table)
    "weather_condition",     # Encoded weather condition (from orders table)
    "speed_multiplier",      # Driver speed multiplier (from drivers table)
    "hour_of_day",           # Hour of day when order was placed
    "is_weekend",            # Whether order was placed on a weekend
    "is_peak_hour",          # Whether order was placed during peak hours
    "experience_level",      # Encoded driver experience level (from drivers table)
    "total_deliveries",      # Driver's total completed deliveries (from drivers table)
]

# Features that require aggregation from historical data
AGGREGATED_FEATURES = [
    name for name, defn in FEATURE_DEFINITIONS.items() if defn.aggregation != AggregationType.NONE
]

# Features computed directly from the order
ORDER_FEATURES = [
    name for name, defn in FEATURE_DEFINITIONS.items() if defn.entity == Entity.ORDER
]
