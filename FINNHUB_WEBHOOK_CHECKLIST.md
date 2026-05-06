# 🚀 Finnhub Webhook Setup Checklist - No Domain Required

## Phase 1: Local Testing ✅
- [ ] API server can start on port 8000
  ```bash
  cd /Users/software/Documents/Stock\ App
  .venv/bin/python -m uvicorn app.main:app --reload --port 8000
  ```

- [ ] Local webhook tests pass
  ```bash
  .venv/bin/python test_finnhub_webhook.py
  ```
  Expected: All 6 tests should show ✅ PASS

- [ ] Health endpoint works
  ```bash
  curl http://localhost:8000/webhooks/finnhub/health
  ```
  Expected: `{"status":"active","webhook_configured":true}`

## Phase 2: Setup Tunnel 🌐

Choose ONE option:

### Option A: ngrok (Easiest)
- [ ] Install ngrok
  ```bash
  brew install ngrok
  ```

- [ ] Start tunnel
  ```bash
  ngrok http 8000
  ```
  Save URL like: `https://abc123.ngrok.io`

- [ ] Verify tunnel works
  ```bash
  curl https://abc123.ngrok.io/webhooks/finnhub/health
  ```

### Option B: Cloudflare Tunnel (Best)
- [ ] Install cloudflared
  ```bash
  brew install cloudflare/cloudflare/cloudflared
  ```

- [ ] Authenticate
  ```bash
  cloudflared tunnel login
  ```

- [ ] Create tunnel
  ```bash
  cloudflared tunnel create stock-webhook
  cloudflared tunnel run stock-webhook
  ```

### Option C: localtunnel (Simple)
- [ ] Install localtunnel
  ```bash
  npm install -g localtunnel
  ```

- [ ] Start tunnel
  ```bash
  lt --port 8000 --subdomain stockapp
  ```
  URL: `https://stockapp.loca.lt`

## Phase 3: Configure Finnhub Dashboard 🔧
- [ ] Go to https://finnhub.io and log in

- [ ] Navigate to Settings → Webhooks

- [ ] Click "Add Webhook"

- [ ] Enter Webhook URL
  ```
  https://your-tunnel-url/webhooks/finnhub/events
  ```

- [ ] Enter Secret
  ```
  d7jkg4pr01qhf13f7jng
  ```

- [ ] Select Event Types
  - [ ] Company News
  - [ ] Market Alert
  - [ ] Price Target
  - [ ] Insider Transaction
  - [ ] Earnings

- [ ] Save Webhook

## Phase 4: Test Integration 🧪
- [ ] Finnhub: Click "Send Test Event"

- [ ] Finnhub: Select "Company News"

- [ ] Check API server logs for webhook received:
  ```bash
  tail -f logs/app.log | grep finnhub
  ```
  Expected: `finnhub_webhook_received` log entry

- [ ] Verify event type appears:
  ```
  finnhub_news_event_processed headline="..." source="Reuters"
  ```

- [ ] Try other event types one by one:
  - [ ] Market Alert
  - [ ] Price Target
  - [ ] Insider Transaction
  - [ ] Earnings

## Phase 5: Monitor Webhook Events 📊
- [ ] View all webhook events
  ```bash
  grep "finnhub_webhook_received" logs/app.log
  ```

- [ ] View specific event type
  ```bash
  grep "finnhub_news_event_processed" logs/app.log
  ```

- [ ] Real-time monitoring
  ```bash
  tail -f logs/app.log | grep finnhub
  ```

## Phase 6: Production Migration 🚀 (When you get a domain)
- [ ] Register domain
- [ ] Set up HTTPS/SSL certificate (Let's Encrypt)
- [ ] Deploy API to production server
- [ ] Update Finnhub webhook URL
  ```
  https://your-domain.com/webhooks/finnhub/events
  ```
- [ ] Remove tunnel
- [ ] Test production webhook

## 📋 Troubleshooting Checklist

### "Connection refused"
- [ ] API server running on port 8000?
- [ ] Tunnel is active and forwarding?
- [ ] Using HTTPS (not HTTP) in Finnhub?

### "Port 8000 already in use"
```bash
lsof -i :8000
kill -9 <PID>
```

### "Signature verification failed"
- [ ] Check FINNHUB_WEBHOOK_SECRET in .env:
  ```bash
  grep FINNHUB_WEBHOOK_SECRET .env
  ```
- [ ] Verify it matches Finnhub dashboard: `d7jkg4pr01qhf13f7jng`

### "Tunnel URL not working"
- [ ] Verify tunnel shows "Forwarding" in terminal
- [ ] ngrok free: URL changes on restart (update Finnhub)
- [ ] Test manually: `curl https://your-url/webhooks/finnhub/health`

### "No logs appearing"
- [ ] Check logs directory exists:
  ```bash
  ls -la /Users/software/Documents/Stock\ App/logs/
  ```
- [ ] Verify webhook is enabled in Finnhub dashboard
- [ ] Check for signature verification errors:
  ```bash
  grep "signature verification" logs/app.log
  ```

## 📚 Documentation Files

- **Quick Start:** `FINNHUB_WEBHOOK_QUICK_START.md`
- **Full Setup:** `FINNHUB_WEBHOOK_SETUP.md`
- **Local Dev:** `FINNHUB_WEBHOOK_LOCAL_DEV.md`
- **This Checklist:** `FINNHUB_WEBHOOK_CHECKLIST.md`

## 🎯 Quick Terminal Setup

Copy & paste this to set up everything:

```bash
cd /Users/software/Documents/Stock\ App

# Terminal 1: Start API server
.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Start tunnel (in another terminal)
ngrok http 8000

# Terminal 3: Run tests (in another terminal)
.venv/bin/python test_finnhub_webhook.py

# Terminal 4: Monitor events (in another terminal)
tail -f logs/app.log | grep finnhub
```

## 💡 Tips
- Keep API server running at all times
- ngrok free tier URL changes on restart (use Cloudflare for persistent URL)
- Always test locally first with `test_finnhub_webhook.py`
- Check logs frequently: they tell you what's happening
- The secret `d7jkg4pr01qhf13f7jng` is already configured

## ✅ Success Criteria
You're done when:
1. ✅ Tunnel is active (see "Forwarding" in terminal)
2. ✅ Finnhub webhook URL updated with tunnel URL
3. ✅ "Send Test Event" returns webhook events in logs
4. ✅ Can see real Finnhub events (when market opens or news happens)
5. ✅ Logs show `finnhub_*_processed` entries

🎉 **That's it! Your webhook is live locally!**

---

**Time to setup:** 5-10 minutes
**Difficulty:** Easy
**Support:** Check FINNHUB_WEBHOOK_LOCAL_DEV.md for detailed help
