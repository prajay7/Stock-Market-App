# Finnhub Webhook Quick Reference

## 🚀 Quick Start (5 minutes) - Local Development

### 1. Verify Setup ✅
```bash
# Check webhook is configured
curl http://localhost:8000/webhooks/finnhub/health
```

Expected response:
```json
{
  "status": "active",
  "webhook_configured": true,
  "timestamp": "2026-04-21T10:30:00.123456"
}
```

### 2. Test Webhook Locally 🧪
```bash
# Run test suite
cd /Users/software/Documents/Stock\ App
.venv/bin/python test_finnhub_webhook.py
```

### 3. Expose Local Server to Internet 🌐

Since you don't have a domain yet, use a tunneling service to expose your local API:

#### Option A: **ngrok** (Recommended - Easy)
```bash
# Install ngrok (one-time)
brew install ngrok

# Start tunnel (in new terminal)
ngrok http 8000

# You'll see:
# Forwarding    https://abc123.ngrok.io -> http://localhost:8000
```

**Copy the HTTPS URL** (e.g., `https://abc123.ngrok.io`)

#### Option B: **Cloudflare Tunnel** (Free)
```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared

# Create tunnel
cloudflared tunnel create stock-app

# Route it
cloudflared tunnel route dns stock-app stock-app.example.com

# Start tunnel
cloudflared tunnel run stock-app
```

#### Option C: **localtunnel** (Simple)
```bash
# Install globally
npm install -g localtunnel

# Start tunnel
lt --port 8000 --subdomain stockapp

# You'll get: https://stockapp.loca.lt
```

### 4. Configure in Finnhub Dashboard 🔧

**URL to enter:**
```
https://your-tunnel-url/webhooks/finnhub/events
```

Examples:
- ngrok: `https://abc123.ngrok.io/webhooks/finnhub/events`
- localtunnel: `https://stockapp.loca.lt/webhooks/finnhub/events`
- cloudflare: `https://stock-app.example.com/webhooks/finnhub/events`

**Secret (already configured):**
```
d7jkg4pr01qhf13f7jng
```

**Event Types to enable:**
- ☑️ Company News
- ☑️ Market Alert
- ☑️ Price Target
- ☑️ Insider Transaction
- ☑️ Earnings

### 5. Test with Finnhub 🧪
1. Save webhook in Finnhub dashboard
2. Click "**Send Test Event**" button
3. Choose event type (Company News)
4. Check your server logs:
   ```bash
   tail -f logs/app.log | grep finnhub
   ```

## �️ Complete Setup Instructions

### Before You Start
1. Make sure your API server is running:
   ```bash
   cd /Users/software/Documents/Stock\ App
   .venv/bin/python -m uvicorn app.main:app --reload --port 8000
   ```

2. Verify webhook is working locally:
   ```bash
   .venv/bin/python test_finnhub_webhook.py
   ```

3. Set up a tunnel (choose one above) to expose your local server

### Step-by-Step

**Step 1:** Start API server in terminal 1
```bash
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

**Step 2:** Start tunnel in terminal 2 (using ngrok example)
```bash
ngrok http 8000
# Copy the HTTPS URL
```

**Step 3:** Go to Finnhub Dashboard
- Log in to https://finnhub.io
- Settings → Webhooks → Add Webhook

**Step 4:** Configure Webhook
- **URL:** `https://your-tunnel-url/webhooks/finnhub/events`
- **Secret:** `d7jkg4pr01qhf13f7jng`
- **Events:** Select all 5 event types
- **Save**

**Step 5:** Test
- Click "Send Test Event"
- Select "Company News"
- Check your server logs for the event

**Step 6:** Monitor Logs
```bash
# Terminal 3: Watch for webhook events
tail -f logs/app.log | grep finnhub
```

### Webhook Events (POST)
```
POST /webhooks/finnhub/events
Header: X-Finnhub-Secret: <shared-secret>
```

### Health Check (GET)
```
GET /webhooks/finnhub/health
```

## 📊 Event Examples

### Company News
```json
{
  "type": "company_news",
  "symbol": "AAPL",
  "data": [{
    "headline": "Apple announces...",
    "source": "Reuters",
    "url": "https://...",
    "datetime": 1671234567
  }]
}
```

### Market Alert
```json
{
  "type": "market_alert",
  "symbol": "TSLA",
  "data": {
    "type": "price_target_up",
    "price": 245.50,
    "changePercent": 5.2
  }
}
```

### Insider Transaction
```json
{
  "type": "insider_transaction",
  "symbol": "MSFT",
  "data": {
    "personName": "John Doe",
    "transactionType": "purchase",
    "shareCount": 10000,
    "sharePrice": 350.25
  }
}
```

## 🔍 Monitoring

### View All Webhook Events
```bash
grep "finnhub_webhook_received" logs/app.log
```

### View Specific Event Type
```bash
grep "finnhub_news_event_processed" logs/app.log
```

### Real-time Monitoring
```bash
tail -f logs/app.log | grep finnhub
```

## 🔐 Security

✓ **Shared-secret** header verification on every request  
✓ **Constant-time** comparison prevents timing attacks  
✓ **Immediate** 2xx response to prevent endpoint disabling  
✓ **Async** processing after acknowledgment  
✓ **Structured** JSON logging for all events  

## 🛠️ Environment Configuration

**File: .env**
```env
FINNHUB=d7jkg4pr01qhf13f7jm0d7jkg4pr01qhf13f7jmg
FINNHUB_WEBHOOK_SECRET=d7jkg4pr01qhf13f7jng
```

## ⚡ Performance

- **Response Time:** < 100ms (returns immediately after signature verification)
- **Processing:** Asynchronous background task
- **Rate Limit:** Unlimited webhook calls (limited by Finnhub subscription)
- **Retry Policy:** Finnhub retries on non-2xx responses

## 📋 Webhook Status

| Metric | Value |
|--------|-------|
| **Status** | ✅ Active |
| **Endpoint** | `/webhooks/finnhub/events` |
| **Authentication** | Shared secret in `X-Finnhub-Secret` header |
| **Secret Configured** | ✅ Yes |
| **Event Types** | 5 supported |
| **Logging** | ✅ Enabled |

## 🐛 Troubleshooting

### Tunnel not working?
1. Make sure API server is running on port 8000
2. Verify tunnel shows "Forwarding" message
3. Test tunnel manually:
   ```bash
   curl https://your-tunnel-url/webhooks/finnhub/health
   ```

### Webhook not receiving events?
1. ✅ Verify tunnel is active (check terminal for "Forwarding" line)
2. ✅ Check webhook URL in Finnhub dashboard matches your tunnel URL
3. ✅ Ensure secret matches: `d7jkg4pr01qhf13f7jng`
4. ✅ Try "Send Test Event" in Finnhub dashboard first
5. ✅ Check server logs:
   ```bash
   tail -f logs/app.log | grep finnhub
   ```

### Tunnel URL expires?
- **ngrok free:** Tunnel URL changes when you restart (update Finnhub webhook URL)
- **ngrok paid:** Static URL (recommended for production)
- **localtunnel:** Same URL persists
- **cloudflare:** Permanent URL (best option)

### Signature verification failing?
1. Verify secret in .env:
   ```bash
   grep FINNHUB_WEBHOOK_SECRET .env
   ```
2. Make sure it matches Finnhub dashboard: `d7jkg4pr01qhf13f7jng`
3. Check logs:
   ```bash
   grep "signature verification" logs/app.log
   ```

### API server won't start?
```bash
# Kill any process using port 8000
lsof -i :8000
kill -9 <PID>

# Restart
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

## 📚 Full Documentation

See [FINNHUB_WEBHOOK_SETUP.md](FINNHUB_WEBHOOK_SETUP.md) for:
- Detailed configuration
- All event types and schemas
- Advanced testing
- Custom event handlers
- Security best practices

## � When You Get a Domain

Once you have a domain and production server:

1. **Update webhook URL in Finnhub dashboard:**
   ```
   https://your-domain.com/webhooks/finnhub/events
   ```

2. **Configure HTTPS/SSL certificate** (Let's Encrypt recommended)

3. **Disable tunnel** on local machine

4. **Ensure API server is running** on production server:
   ```bash
   # With supervisor or systemd for auto-restart
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Monitor production logs:**
   ```bash
   tail -f /var/log/stock-app/app.log | grep finnhub
   ```

## 🔗 Useful Links

- [Finnhub Webhook Docs](https://finnhub.io/docs/api/rest-webhooks)
- [ngrok Documentation](https://ngrok.com/docs)
- [Cloudflare Tunnel Setup](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [localtunnel GitHub](https://github.com/localtunnel/localtunnel)
- [API Status](https://status.finnhub.io)
- [Support Email](support@finnhub.io)
