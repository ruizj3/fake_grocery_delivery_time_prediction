"""
Integration Example for Grocery Delivery Service

This shows how to integrate the prediction API with your grocery delivery service.
Copy this code into your service and call it when orders are confirmed.
"""

import requests
from typing import Optional


class PredictionAPIClient:
    """Client for the delivery time prediction API."""
    
    def __init__(self, api_url: str = "http://localhost:3000"):
        """Initialize the client.
        
        Args:
            api_url: Base URL of the prediction API
        """
        self.api_url = api_url
        self.batch_endpoint = f"{api_url}/predict/batch"
    
    def predict_delivery_time(
        self,
        order_id: str,
        customer_id: str,
        store_id: str,
        store_latitude: float,
        store_longitude: float,
        delivery_latitude: float,
        delivery_longitude: float,
        total: int,
        quantity: int,
        created_at: str,
    ) -> Optional[dict]:
        """Get delivery time prediction for a single order.
        
        Args:
            order_id: Unique order identifier
            customer_id: Customer ID
            store_id: Store/restaurant ID
            store_latitude: Store location latitude
            store_longitude: Store location longitude
            delivery_latitude: Delivery address latitude
            delivery_longitude: Delivery address longitude
            total: Order total in cents
            quantity: Total number of items
            created_at: Order creation timestamp (ISO format)
        
        Returns:
            Prediction dict or None if failed
        """
        request_data = {
            "orders": [{
                "order_id": order_id,
                "customer_id": customer_id,
                "store_id": store_id,
                "store_latitude": store_latitude,
                "store_longitude": store_longitude,
                "delivery_latitude": delivery_latitude,
                "delivery_longitude": delivery_longitude,
                "total": total,
                "quantity": quantity,
                "created_at": created_at,
            }]
        }
        
        try:
            response = requests.post(
                self.batch_endpoint,
                json=request_data,
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result['successful'] > 0:
                return result['predictions'][0]
            elif result['errors']:
                print(f"Prediction error for {order_id}: {result['errors'][0]['error']}")
            
            return None
            
        except requests.exceptions.Timeout:
            print(f"Prediction API timeout for order {order_id}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Could not connect to prediction API for order {order_id}")
            return None
        except Exception as e:
            print(f"Prediction API error for order {order_id}: {e}")
            return None
    
    def predict_batch(self, orders: list[dict]) -> dict:
        """Get predictions for multiple orders at once.
        
        Args:
            orders: List of order dicts with required fields
        
        Returns:
            Response dict with predictions and errors
        """
        request_data = {"orders": orders}
        
        try:
            response = requests.post(
                self.batch_endpoint,
                json=request_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Batch prediction API error: {e}")
            return {
                "predictions": [],
                "total_orders": len(orders),
                "successful": 0,
                "failed": len(orders),
                "errors": [{"order_id": "all", "error": str(e)}]
            }


# Example usage in your grocery delivery service
def example_integration():
    """Example of how to use this in your service."""
    
    # Initialize the client (do this once at app startup)
    prediction_client = PredictionAPIClient(api_url="http://localhost:3000")
    
    # Example 1: Single order prediction
    # Call this when an order is confirmed
    def on_order_confirmed(order, store):
        """Called when order status changes to 'confirmed'."""
        
        # Calculate total quantity from order items
        total_quantity = sum(item.quantity for item in order.items)
        
        # Get prediction
        prediction = prediction_client.predict_delivery_time(
            order_id=order.order_id,
            customer_id=order.customer_id,
            store_id=order.store_id,
            store_latitude=store.latitude,
            store_longitude=store.longitude,
            delivery_latitude=order.delivery_latitude,
            delivery_longitude=order.delivery_longitude,
            total=order.total,
            quantity=total_quantity,
            created_at=order.created_at.isoformat(),
        )
        
        if prediction:
            estimated_minutes = prediction['predicted_delivery_minutes']
            print(f"Order {order.order_id}: ETA {estimated_minutes:.0f} minutes")
            
            # Optionally: Store the prediction in your database
            # order.predicted_delivery_minutes = estimated_minutes
            # order.prediction_model_version = prediction['model_version']
            # db.save(order)
            
            # Optionally: Send ETA to customer via SMS/email
            # notify_customer(order.customer_id, estimated_minutes)
        else:
            print(f"Could not get prediction for order {order.order_id}")
            # Continue without prediction - not critical
    
    # Example 2: Batch predictions for multiple confirmed orders
    # Useful if you process confirmations in batches
    def process_confirmed_orders_batch(orders, stores_map):
        """Process multiple confirmed orders at once."""
        
        # Prepare batch request
        order_data = []
        for order in orders:
            store = stores_map[order.store_id]
            total_quantity = sum(item.quantity for item in order.items)
            
            order_data.append({
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
            })
        
        # Get predictions
        result = prediction_client.predict_batch(order_data)
        
        print(f"Batch predictions: {result['successful']}/{result['total_orders']} successful")
        
        # Process results
        for prediction in result['predictions']:
            print(f"  Order {prediction['order_id']}: {prediction['predicted_delivery_minutes']:.0f} min")
        
        return result


if __name__ == "__main__":
    print("This is an integration example.")
    print("Copy the PredictionAPIClient class into your grocery delivery service.")
    print("\nThen call prediction_client.predict_delivery_time() when orders are confirmed.")
