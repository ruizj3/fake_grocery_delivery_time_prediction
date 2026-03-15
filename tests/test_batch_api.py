"""
Test script for batch prediction API endpoint.

This demonstrates how your other service should send confirmed orders
to this API for prediction.
"""

import requests
from datetime import datetime


def test_batch_prediction():
    """Test the batch prediction endpoint."""
    
    # API endpoint
    api_url = "http://localhost:3000/predict/batch"
    
    # Example: Your service sends confirmed orders in batches
    confirmed_orders = {
        "orders": [
            {
                "order_id": "test_order_001",
                "customer_id": "customer_123",
                "store_id": "store_456",
                "store_latitude": 47.6062,
                "store_longitude": -122.3321,
                "delivery_latitude": 47.6205,
                "delivery_longitude": -122.3493,
                "total": 4500,  # $45.00 in cents
                "quantity": 5,
                "created_at": datetime.now().isoformat(),
            },
            {
                "order_id": "test_order_002",
                "customer_id": "customer_456",
                "store_id": "store_789",
                "store_latitude": 47.6097,
                "store_longitude": -122.3331,
                "delivery_latitude": 47.6145,
                "delivery_longitude": -122.3281,
                "total": 3200,  # $32.00
                "quantity": 3,
                "created_at": datetime.now().isoformat(),
            },
        ]
    }
    
    print("Sending batch prediction request...")
    print(f"Number of orders: {len(confirmed_orders['orders'])}\n")
    
    try:
        response = requests.post(api_url, json=confirmed_orders)
        response.raise_for_status()
        
        result = response.json()
        
        print("✓ Batch prediction successful!")
        print(f"\nResults:")
        print(f"  Total orders: {result['total_orders']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}\n")
        
        if result['predictions']:
            print("Predictions:")
            for pred in result['predictions']:
                print(f"  Order {pred['order_id']}: {pred['predicted_delivery_minutes']:.1f} minutes")
                print(f"    Model: {pred['model_version'][:10]}...")
                print(f"    Timestamp: {pred['prediction_timestamp']}")
                print()
        
        if result['errors']:
            print("Errors:")
            for error in result['errors']:
                print(f"  Order {error['order_id']}: {error['error']}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        print("\nMake sure the API is running:")
        print("  python -m delivery_ml.serving.api")
        return None
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"\nResponse: {e.response.text}")
        return None


def test_get_recent_predictions():
    """Test getting recent predictions from the API."""
    
    api_url = "http://localhost:3000/predictions/recent?limit=10"
    
    print("\n" + "="*60)
    print("Getting recent predictions...")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n✓ Found {result['count']} recent predictions\n")
        
        if result['predictions']:
            for i, pred in enumerate(result['predictions'][:5], 1):
                print(f"{i}. Order {pred['order_id'][:20]}...")
                print(f"   Prediction: {pred['predicted_delivery_minutes']:.1f} minutes")
                print(f"   Created: {pred['created_at']}")
                print()
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        return None


def test_get_stats():
    """Test getting prediction statistics."""
    
    api_url = "http://localhost:3000/predictions/stats"
    
    print("="*60)
    print("Getting prediction statistics...")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n✓ Prediction Statistics:")
        print(f"  Total predictions: {result['total_predictions']}")
        print(f"  Unique orders: {result['unique_orders']}")
        if result['total_predictions'] > 0:
            print(f"  Average: {result['avg_prediction_minutes']:.1f} minutes")
            print(f"  Range: {result['min_prediction_minutes']:.1f} - {result['max_prediction_minutes']:.1f} minutes")
            print(f"  Time range: {result['earliest_prediction']} to {result['latest_prediction']}")
        print()
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        return None


if __name__ == "__main__":
    print("="*60)
    print("Testing Batch Prediction API")
    print("="*60)
    print()
    
    # Test batch prediction
    result = test_batch_prediction()
    
    if result:
        # Test getting recent predictions
        test_get_recent_predictions()
        
        # Test getting stats
        test_get_stats()
        
        print("="*60)
        print("✓ All tests completed!")
        print("\nIntegration steps for your service:")
        print("1. When an order is confirmed, collect order details")
        print("2. Send POST to /predict/batch with order data")
        print("3. Receive predictions in response")
        print("4. Predictions are automatically stored in predictions.db")
        print("="*60)
