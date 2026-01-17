"""Tests for feature computation."""

from datetime import datetime

import pytest

from delivery_ml.features.computation import haversine_distance_km


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_returns_zero(self):
        """Distance from a point to itself should be zero."""
        distance = haversine_distance_km(47.6, -122.3, 47.6, -122.3)
        assert distance == 0.0

    def test_known_distance(self):
        """Test against a known distance."""
        # Seattle to Portland is approximately 233 km
        seattle = (47.6062, -122.3321)
        portland = (45.5152, -122.6784)

        distance = haversine_distance_km(
            seattle[0], seattle[1], portland[0], portland[1]
        )

        assert 230 < distance < 240  # Allow some tolerance

    def test_symmetry(self):
        """Distance should be symmetric."""
        d1 = haversine_distance_km(47.6, -122.3, 45.5, -122.6)
        d2 = haversine_distance_km(45.5, -122.6, 47.6, -122.3)

        assert abs(d1 - d2) < 0.001


class TestFeatureComputation:
    """Tests for feature computation logic."""

    def test_order_features_hour_extraction(self):
        """Test that hour of day is correctly extracted."""
        from delivery_ml.features.computation import FeatureComputer

        computer = FeatureComputer(":memory:")

        features = computer.compute_order_features(
            restaurant_lat=47.6,
            restaurant_lon=-122.3,
            delivery_lat=47.65,
            delivery_lon=-122.35,
            placed_at=datetime(2024, 3, 15, 14, 30),  # 2:30 PM
            order_total_cents=2500,
            item_count=3,
        )

        assert features["hour_of_day"] == 14
        assert features["day_of_week"] == 4  # Friday
        assert features["is_weekend"] is False
        assert features["total"] == 2500
        assert features["quantity"] == 3

    def test_weekend_detection(self):
        """Test weekend flag."""
        from delivery_ml.features.computation import FeatureComputer

        computer = FeatureComputer(":memory:")

        # Saturday
        features_sat = computer.compute_order_features(
            restaurant_lat=47.6,
            restaurant_lon=-122.3,
            delivery_lat=47.65,
            delivery_lon=-122.35,
            placed_at=datetime(2024, 3, 16, 12, 0),  # Saturday
            order_total_cents=2500,
            item_count=3,
        )

        # Monday
        features_mon = computer.compute_order_features(
            restaurant_lat=47.6,
            restaurant_lon=-122.3,
            delivery_lat=47.65,
            delivery_lon=-122.35,
            placed_at=datetime(2024, 3, 18, 12, 0),  # Monday
            order_total_cents=2500,
            item_count=3,
        )

        assert features_sat["is_weekend"] is True
        assert features_mon["is_weekend"] is False
