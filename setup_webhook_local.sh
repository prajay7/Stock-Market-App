#!/bin/bash
# Quick setup script for Finnhub webhook testing locally with ngrok
# Usage: bash setup_webhook_local.sh

set -e

PROJECT_PATH="/Users/software/Documents/Stock App"
cd "$PROJECT_PATH"

echo "════════════════════════════════════════════════════════"
echo "🚀 Finnhub Webhook - Local Development Setup"
echo "════════════════════════════════════════════════════════"

# Check if ngrok is installed
echo ""
echo "📋 Step 1: Checking prerequisites..."
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Installing..."
    brew install ngrok
else
    echo "✅ ngrok is installed"
fi

# Check Python environment
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi
echo "✅ Python environment is ready"

# Check webhook configuration
if ! grep -q "FINNHUB_WEBHOOK_SECRET" .env; then
    echo "❌ FINNHUB_WEBHOOK_SECRET not configured in .env"
    exit 1
fi
echo "✅ Webhook secret is configured"

# Verify webhook files
echo ""
echo "📋 Step 2: Verifying webhook files..."
if [ ! -f "app/api/routes_webhooks.py" ]; then
    echo "❌ Webhook routes not found"
    exit 1
fi
echo "✅ Webhook routes configured"

if [ ! -f "test_finnhub_webhook.py" ]; then
    echo "❌ Test script not found"
    exit 1
fi
echo "✅ Test script available"

# Run syntax checks
echo ""
echo "📋 Step 3: Running syntax validation..."
.venv/bin/python -m py_compile app/api/routes_webhooks.py app/main.py
echo "✅ All files compile successfully"

# Run local tests
echo ""
echo "📋 Step 4: Running local webhook tests..."
echo "(This verifies webhook works on localhost:8000)"
echo ""
.venv/bin/python test_finnhub_webhook.py
echo ""

# Show configuration
echo "════════════════════════════════════════════════════════"
echo "✅ SETUP COMPLETE"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📝 Configuration:"
echo "   Webhook Secret: $(grep FINNHUB_WEBHOOK_SECRET .env | cut -d= -f2)"
echo "   API Port: 8000"
echo "   Event Endpoint: /webhooks/finnhub/events"
echo "   Health Check: /webhooks/finnhub/health"
echo ""
echo "🚀 Quick Start (3 Steps):"
echo ""
echo "1️⃣  Terminal 1: Start API Server"
echo "    .venv/bin/python -m uvicorn app.main:app --reload --port 8000"
echo ""
echo "2️⃣  Terminal 2: Start Tunnel (copy the URL)"
echo "    ngrok http 8000"
echo ""
echo "3️⃣  Browser: Configure in Finnhub Dashboard"
echo "    https://finnhub.io → Settings → Webhooks → Add"
echo "    - URL: https://YOUR_NGROK_URL/webhooks/finnhub/events"
echo "    - Secret: $(grep FINNHUB_WEBHOOK_SECRET .env | cut -d= -f2)"
echo ""
echo "📚 More info:"
echo "   cat FINNHUB_WEBHOOK_LOCAL_DEV.md"
echo ""
echo "════════════════════════════════════════════════════════"
