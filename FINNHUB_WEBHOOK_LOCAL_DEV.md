# Local Development: Finnhub Webhook Testing Without a Domain

Since you don't have a domain yet, use a **tunneling service** to expose your local development server to the internet. This allows Finnhub to send webhook events to your local machine.

## 🌐 Tunneling Options

### Option 1: **ngrok** (Easiest, Recommended for Testing)

**Installation:**
```bash
brew install ngrok
```

**Usage:**
```bash
# Terminal 1: Start your API server
cd /Users/software/Documents/Stock\ App
.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Start ngrok tunnel
ngrok http 8000

# You'll see output like:
# Forwarding    https://abc123.ngrok.io -> http://localhost:8000
```

**Pros:**
- ✅ Easiest to set up
- ✅ Works immediately
- ✅ Good for testing
- ✅ Free tier available

**Cons:**
- ❌ URL changes on restart (free tier)
- ❌ Limited bandwidth

**For Finnhub:** Use `https://abc123.ngrok.io/webhooks/finnhub/events`

---

### Option 2: **Cloudflare Tunnel** (Best, Free, Permanent URL)

**Installation:**
```bash
brew install cloudflare/cloudflare/cloudflared
```

**Setup:**
```bash
# Authenticate with Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create stock-webhook

# Route DNS (if you have a Cloudflare domain)
cloudflared tunnel route dns stock-webhook stock-webhook.your-domain.com

# Run tunnel
cloudflared tunnel run stock-webhook

# To stop
# Press Ctrl+C
```

**Pros:**
- ✅ Permanent URL (no need to update Finnhub)
- ✅ Free with Cloudflare
- ✅ Good security
- ✅ Best for production-like setup

**Cons:**
- ⚠️ Requires Cloudflare account
- ⚠️ Slightly more setup

**For Finnhub:** Use `https://stock-webhook.your-domain.com/webhooks/finnhub/events`

---

### Option 3: **localtunnel** (Simple, Persistent)

**Installation:**
```bash
npm install -g localtunnel
```

**Usage:**
```bash
# Terminal 1: Start API server
cd /Users/software/Documents/Stock\ App
.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Start tunnel
lt --port 8000 --subdomain stockapp

# You'll get: https://stockapp.loca.lt
```

**Pros:**
- ✅ Very simple
- ✅ Free
- ✅ Predictable URL
- ✅ Good for testing

**Cons:**
- ❌ May go down
- ❌ Not as reliable as others

**For Finnhub:** Use `https://stockapp.loca.lt/webhooks/finnhub/events`

---

## 🚀 Complete Setup Example (Using ngrok)

### Step 1: Start Your API

```bash
cd /Users/software/Documents/Stock\ App

# Terminal 1
.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Output should show:
# Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start Tunnel

```bash
# Terminal 2 (new)
ngrok http 8000

# You'll see:
# Session Status                online
# Account                       your-email@example.com
# Version                       3.0.0
# Forwarding                    https://abc123def456.ngrok.io -> http://localhost:8000
```

**Copy the HTTPS URL** (e.g., `https://abc123def456.ngrok.io`)

### Step 3: Verify Everything Works

```bash
# Terminal 3 (new)
curl https://abc123def456.ngrok.io/webhooks/finnhub/health

# Should return:
# {"status":"active","webhook_configured":true,"timestamp":"2026-04-21T..."}
```

### Step 4: Test Webhook Locally

```bash
# Terminal 3
cd /Users/software/Documents/Stock\ App
.venv/bin/python test_finnhub_webhook.py

# Should see:
# ✅ PASS - Health Check
# ✅ PASS - Company News
# ✅ PASS - Market Alert
# etc.
```

### Step 5: Configure in Finnhub Dashboard

1. Go to **https://finnhub.io**
2. Settings → **Webhooks** → **+ Add Webhook**
3. Fill in:
   - **URL:** `https://abc123def456.ngrok.io/webhooks/finnhub/events`
   - **Secret:** `d7jkg4pr01qhf13f7jng`
   - **Events:** Select all 5 types
4. Click **Save**

### Step 6: Test with Finnhub

1. In Finnhub webhook config, click **"Send Test Event"**
2. Select **"Company News"**
3. Watch your Terminal 1 (API server):

```
INFO:     127.0.0.1:12345 - "POST /webhooks/finnhub/events HTTP/1.1" 200
finnhub_webhook_received event_type="company_news" symbol="AAPL"
finnhub_news_event_processed headline="Apple announces..."
```

### Step 7: Monitor Events

```bash
# Terminal 4 (new) - Real-time webhook monitoring
cd /Users/software/Documents/Stock\ App
tail -f logs/app.log | grep finnhub
```

---

## ⚡ Important Notes

### About ngrok Free Tier

The free version gives you a **new random URL every time** you restart ngrok. 

**What to do:**
1. Get the new URL from ngrok terminal
2. Update it in Finnhub dashboard
3. Test again

**To get a permanent URL:**
- Pay for ngrok Pro ($12/month)
- Use Cloudflare Tunnel instead (free)

### File Structure

```
/Users/software/Documents/Stock App/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── api/
│   │   └── routes_webhooks.py  # Webhook handler
│   └── core/
│       └── config.py           # Config with FINNHUB settings
├── .env                        # Contains FINNHUB_WEBHOOK_SECRET
└── test_finnhub_webhook.py     # Test script
```

### Environment Variables

Make sure `.env` has:
```env
FINNHUB=d7jkg4pr01qhf13f7jm0d7jkg4pr01qhf13f7jmg
FINNHUB_WEBHOOK_SECRET=d7jkg4pr01qhf13f7jng
```

---

## 🐛 Common Issues

### "Port 8000 already in use"

```bash
# Find and kill the process
lsof -i :8000
kill -9 <PID>

# Start again
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

### "Connection refused"

Make sure:
1. API server is running (check Terminal 1)
2. Tunnel is active (check Terminal 2 for "Forwarding" line)
3. Use HTTPS in Finnhub (not HTTP)

### "Signature verification failed"

Check that `FINNHUB_WEBHOOK_SECRET` in `.env` matches Finnhub dashboard:
```bash
grep FINNHUB_WEBHOOK_SECRET /Users/software/Documents/Stock\ App/.env
# Should show: d7jkg4pr01qhf13f7jng
```

### ngrok URL works locally but not from Finnhub

1. Verify URL in ngrok terminal (check for "Forwarding" line)
2. Test manually:
   ```bash
   curl https://your-ngrok-url/webhooks/finnhub/health
   ```
3. If it works locally but not from Finnhub, check:
   - Firewall settings
   - Exact URL format (no trailing slash)
   - Secret is correct

---

## 📊 Testing Workflow

```
Terminal 1: API Server    │  Terminal 2: Tunnel       │  Terminal 3: Monitoring
                          │                           │
.venv/bin/python -m       │ ngrok http 8000           │ tail -f logs/app.log | 
  uvicorn app.main:app    │                           │ grep finnhub
  --reload --port 8000    │ Forwarding:               │
                          │ https://abc123.ngrok.io   │ [Watch logs here]
[Running...]              │                           │
                          │ Status: Online            │
```

1. **Terminal 1:** API listens on `http://localhost:8000`
2. **Terminal 2:** Tunnel exposes to `https://abc123.ngrok.io`
3. **Terminal 3:** Monitor webhook events in logs

---

## 🎯 Next Steps

1. ✅ Choose a tunnel method (ngrok recommended for testing)
2. ✅ Start API server
3. ✅ Start tunnel
4. ✅ Run local tests: `python test_finnhub_webhook.py`
5. ✅ Configure webhook in Finnhub dashboard
6. ✅ Send test event from Finnhub
7. ✅ Watch logs for webhook events
8. ✅ Build your event handlers

---

## 📚 References

- **ngrok:** https://ngrok.com/docs
- **Cloudflare Tunnel:** https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- **localtunnel:** https://github.com/localtunnel/localtunnel
- **Finnhub Webhooks:** https://finnhub.io/docs/api/rest-webhooks
- **FastAPI Docs:** https://fastapi.tiangolo.com/

---

## 💡 Tips

- Keep a terminal **always open with the API server** for development
- Restart ngrok once a day (refresh tunnel URL) to keep latency low
- Use Cloudflare Tunnel for longer sessions (no URL changes)
- Test locally with `test_finnhub_webhook.py` before configuring Finnhub
- Check logs frequently to understand event flow

Need help? Check [FINNHUB_WEBHOOK_SETUP.md](FINNHUB_WEBHOOK_SETUP.md) for full documentation.
