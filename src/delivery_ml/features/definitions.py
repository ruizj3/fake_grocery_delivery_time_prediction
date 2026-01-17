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


# -----------------------------------------------------------------------------
# Feature Sets (groups of features used together)
# -----------------------------------------------------------------------------

TRAINING_FEATURES = [
    "restaurant_avg_delivery_minutes_30d",
    "restaurant_order_count_30d",
    "customer_order_count_30d",
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "total",
    "quantity",
]

# Features that require aggregation from historical data
AGGREGATED_FEATURES = [
    name for name, defn in FEATURE_DEFINITIONS.items() if defn.aggregation != AggregationType.NONE
]

# Features computed directly from the order
ORDER_FEATURES = [
    name for name, defn in FEATURE_DEFINITIONS.items() if defn.entity == Entity.ORDER
]
