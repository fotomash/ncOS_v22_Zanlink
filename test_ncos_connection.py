#!/usr/bin/env python3
"""
Test client for ncOS Local Server
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "https://emerging-tiger-fair.ngrok-free.app"
LOCAL_URL = "http://localhost:8000"

# Headers to skip ngrok warning
headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true"
}

def test_endpoint(url, endpoint, method="GET", data=None):
    """Test an endpoint"""
    full_url = f"{url}{endpoint}"
    print(f"\nTesting {method} {full_url}")

    try:
        if method == "GET":
            response = requests.get(full_url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(full_url, json=data, headers=headers, timeout=10)

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Connection failed - make sure the server is running")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """Run tests"""
    print("ncOS Local Server Test Client")
    print("=" * 50)

    # Test local connection first
    print("\nTesting local connection...")
    test_endpoint(LOCAL_URL, "/")
    test_endpoint(LOCAL_URL, "/health")

    # Test ngrok connection
    print("\n\nTesting ngrok connection...")
    test_endpoint(BASE_URL, "/")
    test_endpoint(BASE_URL, "/health")

    # Test data endpoints
    print("\n\nTesting data endpoints...")
    test_endpoint(BASE_URL, "/data")

    # Test trading API
    print("\n\nTesting trading API...")
    test_trade = {
        "symbol": "XAUUSD",
        "action": "BUY",
        "quantity": 0.01,
        "price": 3357.50
    }
    test_endpoint(BASE_URL, "/api/trading", method="POST", data=test_trade)

    # Test market data
    print("\n\nTesting market data...")
    test_endpoint(BASE_URL, "/api/market")

    # Test signals
    print("\n\nTesting signals...")
    test_signal = {
        "symbol": "XAUUSD",
        "type": "BUY",
        "strength": 0.75,
        "reason": "Test signal"
    }
    test_endpoint(BASE_URL, "/api/signals", method="POST", data=test_signal)

if __name__ == "__main__":
    main()
