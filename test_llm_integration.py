#!/usr/bin/env python3
"""
Quick test script for ncOS LLM Integration
"""

import requests
import json
from datetime import datetime

def test_integration():
    """Test the integration components"""

    print("🧪 Testing ncOS LLM Integration...")
    print("=" * 50)

    # Test 1: Quick Status
    print("\n1️⃣ Testing Quick Status...")
    try:
        response = requests.get("http://localhost:8081/api/v1/quick/status?symbol=XAUUSD")
        if response.status_code == 200:
            print("✅ Quick Status: SUCCESS")
            print(f"   Response: {response.json()['one_line_summary']}")
        else:
            print("❌ Quick Status: FAILED")
    except Exception as e:
        print(f"❌ Quick Status: ERROR - {e}")

    # Test 2: Pattern Detection
    print("\n2️⃣ Testing Pattern Detection...")
    try:
        response = requests.get("http://localhost:8081/api/v1/patterns/detect?symbol=XAUUSD")
        if response.status_code == 200:
            data = response.json()
            print("✅ Pattern Detection: SUCCESS")
            print(f"   Patterns found: {len(data['patterns'])}")
            print(f"   Bias: {data['bias']}")
        else:
            print("❌ Pattern Detection: FAILED")
    except Exception as e:
        print(f"❌ Pattern Detection: ERROR - {e}")

    # Test 3: Bridge Process
    print("\n3️⃣ Testing Integration Bridge...")
    try:
        response = requests.post("http://localhost:8003/bridge/process", json={
            "action": "market_overview",
            "context": {"symbol": "XAUUSD"},
            "format": "chatgpt"
        })
        if response.status_code == 200:
            print("✅ Integration Bridge: SUCCESS")
            print("   Response preview:")
            print(response.json()["response"][:200] + "...")
        else:
            print("❌ Integration Bridge: FAILED")
    except Exception as e:
        print(f"❌ Integration Bridge: ERROR - {e}")

    # Test 4: Prompt Generation
    print("\n4️⃣ Testing Prompt Generation...")
    try:
        response = requests.get(
            "http://localhost:8003/bridge/prompt",
            params={"query": "Should I go long on Gold?", "symbol": "XAUUSD"}
        )
        if response.status_code == 200:
            print("✅ Prompt Generation: SUCCESS")
            print("   Generated prompt length:", len(response.json()["prompt"]))
        else:
            print("❌ Prompt Generation: FAILED")
    except Exception as e:
        print(f"❌ Prompt Generation: ERROR - {e}")

    print("\n" + "=" * 50)
    print("🏁 Test Complete!")

if __name__ == "__main__":
    test_integration()
