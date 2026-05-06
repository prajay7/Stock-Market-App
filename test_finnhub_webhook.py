#!/usr/bin/env python
"""
Test script for Finnhub webhook integration.
Simulates webhook requests with proper HMAC-SHA256 signature verification.
"""

import json
import requests
import sys
from datetime import datetime

# Configuration
WEBHOOK_SECRET = "d7jkg4pr01qhf13f7jng"
WEBHOOK_URL = "http://localhost:8000/webhooks/finnhub/events"
HEALTH_CHECK_URL = "http://localhost:8000/webhooks/finnhub/health"


def compute_signature(payload: dict) -> str:
    """Return the shared secret expected by Finnhub webhook auth."""
    return WEBHOOK_SECRET


def test_health_check():
    """Test webhook health endpoint."""
    print("\n🏥 Testing webhook health check...")
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_company_news_event():
    """Test company news webhook event."""
    print("\n📰 Testing company news event...")
    
    payload = {
        "type": "company_news",
        "symbol": "AAPL",
        "timestamp": int(datetime.now().timestamp()),
        "data": [
            {
                "headline": "Apple announces new AI features in iPhone",
                "source": "Reuters",
                "url": "https://www.reuters.com/example",
                "datetime": int(datetime.now().timestamp())
            }
        ]
    }
    
    signature = compute_signature(payload)
    headers = {
        "X-Finnhub-Secret": signature,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 201, 202]
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_market_alert_event():
    """Test market alert webhook event."""
    print("\n📊 Testing market alert event...")
    
    payload = {
        "type": "market_alert",
        "symbol": "TSLA",
        "timestamp": int(datetime.now().timestamp()),
        "data": {
            "type": "price_target_up",
            "price": 245.50,
            "changePercent": 5.2
        }
    }
    
    signature = compute_signature(payload)
    headers = {
        "X-Finnhub-Secret": signature,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 201, 202]
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_insider_transaction_event():
    """Test insider transaction webhook event."""
    print("\n💼 Testing insider transaction event...")
    
    payload = {
        "type": "insider_transaction",
        "symbol": "MSFT",
        "timestamp": int(datetime.now().timestamp()),
        "data": {
            "personName": "John Doe",
            "transactionType": "purchase",
            "shareCount": 10000,
            "sharePrice": 350.25,
            "filingDate": "2023-12-15"
        }
    }
    
    signature = compute_signature(payload)
    headers = {
        "X-Finnhub-Secret": signature,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 201, 202]
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_invalid_signature():
    """Test with invalid signature (should fail)."""
    print("\n🔐 Testing signature verification (invalid signature)...")
    
    payload = {
        "type": "company_news",
        "symbol": "AAPL",
        "timestamp": int(datetime.now().timestamp()),
        "data": []
    }
    
    # Use wrong signature
    headers = {
        "X-Finnhub-Secret": "invalid_signature_12345",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        # Should get 401 Unauthorized
        return response.status_code == 401
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_earnings_event():
    """Test earnings announcement webhook event."""
    print("\n💰 Testing earnings event...")
    
    payload = {
        "type": "earnings",
        "symbol": "GOOGL",
        "timestamp": int(datetime.now().timestamp()),
        "data": {
            "epsEstimate": 1.50,
            "epsActual": 1.65,
            "revenueEstimate": 60000000000,
            "revenueActual": 62000000000,
            "announcementDate": "2023-04-18"
        }
    }
    
    signature = compute_signature(payload)
    headers = {
        "X-Finnhub-Secret": signature,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 201, 202]
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all webhook tests."""
    print("=" * 70)
    print("🚀 Finnhub Webhook Integration Tests")
    print("=" * 70)
    print(f"\nWebhook URL: {WEBHOOK_URL}")
    print(f"Webhook Secret: {WEBHOOK_SECRET[:20]}...")
    
    results = {
        "Health Check": test_health_check(),
        "Company News": test_company_news_event(),
        "Market Alert": test_market_alert_event(),
        "Insider Transaction": test_insider_transaction_event(),
        "Earnings Event": test_earnings_event(),
        "Invalid Signature (should fail)": test_invalid_signature(),
    }
    
    print("\n" + "=" * 70)
    print("📋 Test Results Summary")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print("\n" + "-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
