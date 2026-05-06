# Finnhub Webhook Integration Setup Guide

## Overview

This project includes a complete Finnhub webhook integration that receives real-time events for:
- **Company News** - Updates on latest news articles for watched stocks
- **Market Alerts** - Price movements and significant market events
- **Price Targets** - Analyst price target updates
- **Insider Transactions** - SEC insider trading filings
- **Earnings Announcements** - Quarterly earnings and guidance

## Configuration

### 1. Environment Variables

The following environment variables are already configured in `.env`:

```env
FINNHUB=<your_finnhub_api_key>
FINNHUB_WEBHOOK_SECRET=d7jkg4pr01qhf13f7jng
```

Replace `<your_finnhub_api_key>` with your actual Finnhub API key.

### 2. Webhook Endpoint

The webhook endpoint is available at:

```
POST http://your-domain:8000/webhooks/finnhub/events
```

### 3. Webhook Health Check

To verify the webhook is configured and active:

```
GET http://your-domain:8000/webhooks/finnhub/health
```

Response:
```json
{
  "status": "active",
  "webhook_configured": true,
  "timestamp": "2026-04-21T10:30:00.123456"
}
```

## Setting up Webhook in Finnhub Dashboard

### Step 1: Access Finnhub Dashboard
1. Go to https://finnhub.io
2. Log in to your account
3. Navigate to Settings → Webhooks

### Step 2: Configure Webhook Endpoint

**Webhook URL:**
```
https://your-domain.com:8000/webhooks/finnhub/events
```

**Replace with:**
- `your-domain.com` - Your server's domain name or IP address
- `8000` - API port (or whatever port you're running the API on)

### Step 3: Set Webhook Secret

**Secret:**
```
d7jkg4pr01qhf13f7jng
```

This is already configured in your `.env` file as `FINNHUB_WEBHOOK_SECRET`. Finnhub will send this shared secret in the `X-Finnhub-Secret` header for webhook authentication.

### Step 4: Select Event Types

Choose which event types you want to receive:

- ✅ **Company News** - Latest news articles per company
- ✅ **Market Alert** - Price movements and significant events
- ✅ **Price Target** - Analyst price targets and ratings
- ✅ **Insider Transaction** - Insider trading activities
- ✅ **Earnings** - Earnings announcements and results

### Step 5: Customize Events (Optional)

You can set filters for specific:
- Symbols or watchlists
- News sources
- Price alert thresholds
- Event importance levels

### Step 6: Save and Test

1. Click "Save"
2. Finnhub provides a test button to send a sample event
3. Check your server logs for the webhook event

## Webhook Request Format

### Headers

Every Finnhub webhook request includes:

```
X-Finnhub-Secret: d7jkg4pr01qhf13f7jng
Content-Type: application/json
```

The `X-Finnhub-Secret` header is verified against your configured shared secret to ensure authenticity.

### Example Request Body

**Company News Event:**
```json
{
  "type": "company_news",
  "symbol": "AAPL",
  "timestamp": 1671234567,
  "data": [
    {
      "headline": "Apple announces new product",
      "source": "Reuters",
      "url": "https://...",
      "datetime": 1671234567
    }
  ]
}
```

**Market Alert Event:**
```json
{
  "type": "market_alert",
  "symbol": "TSLA",
  "timestamp": 1671234567,
  "data": {
    "type": "price_target_up",
    "price": 245.50,
    "changePercent": 5.2
  }
}
```

**Insider Transaction Event:**
```json
{
  "type": "insider_transaction",
  "symbol": "MSFT",
  "timestamp": 1671234567,
  "data": {
    "personName": "John Doe",
    "transactionType": "purchase",
    "shareCount": 10000,
    "sharePrice": 350.25,
    "filingDate": "2023-12-15"
  }
}
```

## Event Processing

### Event Flow

1. **Receive** - Webhook endpoint receives HTTPS POST request
2. **Verify** - HMAC-SHA256 signature verified using `FINNHUB_WEBHOOK_SECRET`
3. **Acknowledge** - Return 2xx status code immediately (prevents timeouts)
4. **Process** - Event data processed asynchronously:
   - Logged with structured JSON logging
   - Stored in database (when DB integration added)
   - Triggers relevant business logic:
     - News sentiment analysis
     - Trading alerts
     - Price target updates
     - Insider transaction analysis

### Logging

All webhook events are logged with structured JSON format in `app/api/routes_webhooks.py`:

```python
logger.info(
    "finnhub_webhook_received",
    extra={
        "event_type": "company_news",
        "symbol": "AAPL",
        "timestamp": 1671234567,
    }
)
```

Check logs with:
```bash
# View logs
tail -f logs/app.log | grep finnhub

# Filter by event type
grep "finnhub_webhook_received" logs/app.log
```

## Event Handlers

The webhook supports the following event types through specific handler functions:

### 1. Company News Events
- **Handler:** `_process_company_news_event()`
- **Data:** Headline, source, URL, publish datetime
- **Action:** Log news, trigger sentiment analysis, store in DB

### 2. Market Alert Events
- **Handler:** `_process_market_alert_event()`
- **Data:** Alert type, price, change percentage
- **Action:** Trigger trading alerts, notify users

### 3. Price Target Events
- **Handler:** `_process_price_target_event()`
- **Data:** Target price, analyst name, rating
- **Action:** Update forecasts, track analyst consensus

### 4. Insider Transaction Events
- **Handler:** `_process_insider_transaction_event()`
- **Data:** Person name, transaction type, share count, price
- **Action:** Analyze insider activity, flag significant transactions

### 5. Earnings Events
- **Handler:** `_process_earnings_event()`
- **Data:** EPS estimates/actuals, revenue estimates/actuals
- **Action:** Update forecasts, compare vs expectations

## Error Handling

### Signature Verification Fails
- **Response:** 401 Unauthorized
- **Log:** `"Finnhub webhook signature verification failed"`
- **Action:** Check that `FINNHUB_WEBHOOK_SECRET` matches Finnhub configuration

### Invalid Event Type
- **Response:** 200 OK (still acknowledged)
- **Log:** `"Unknown Finnhub event type: {type}"`
- **Action:** May indicate new Finnhub event type - add handler

### Processing Error
- **Response:** 200 OK (still acknowledged to prevent retries)
- **Log:** `"Error processing Finnhub webhook: {error}"`
- **Action:** Check server logs for details

### Endpoint Disabled

Finnhub automatically disables your webhook if it fails to receive 2xx responses for consecutive days. To re-enable:

1. Fix the issue (check logs)
2. Contact Finnhub support or re-add the webhook
3. Test with "Send Test Event" button

## Testing the Webhook

### Method 1: Finnhub Dashboard Test

1. In Finnhub dashboard, find your webhook configuration
2. Click "Send Test Event"
3. Select event type (News, Alert, etc.)
4. Check server logs for received event

### Method 2: Manual cURL Test

```bash
# Set variables
WEBHOOK_SECRET="d7jkg4pr01qhf13f7jng"
WEBHOOK_URL="http://localhost:8000/webhooks/finnhub/events"

# Create test payload
PAYLOAD='{"type":"company_news","symbol":"AAPL","timestamp":1671234567,"data":[{"headline":"Test","source":"Test","url":"http://test.com"}]}'

# Use the shared secret directly
SIGNATURE="$WEBHOOK_SECRET"

# Send request
curl -X POST "$WEBHOOK_URL" \
   -H "X-Finnhub-Secret: $SIGNATURE" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"
```

### Method 3: Python Test Script

```python
import requests
import hmac
import hashlib
import json

webhook_secret = "d7jkg4pr01qhf13f7jng"
webhook_url = "http://localhost:8000/webhooks/finnhub/events"

payload = {
    "type": "company_news",
    "symbol": "AAPL",
    "timestamp": 1671234567,
    "data": [{
        "headline": "Test news from Finnhub",
        "source": "Reuters",
        "url": "https://example.com",
        "datetime": 1671234567
    }]
}

body = json.dumps(payload).encode()
signature = hmac.new(
    webhook_secret.encode(),
    body,
    hashlib.sha256
).hexdigest()

headers = {
   "X-Finnhub-Secret": signature,
    "Content-Type": "application/json"
}

response = requests.post(webhook_url, json=payload, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

## Monitoring and Logging

### View All Webhook Events
```bash
grep "finnhub_webhook_received" logs/app.log
```

### View Specific Event Type
```bash
grep "finnhub_news_event_processed" logs/app.log
```

### Check Webhook Health
```bash
curl http://localhost:8000/webhooks/finnhub/health
```

### Real-time Log Monitoring
```bash
tail -f logs/app.log | grep finnhub
```

## Webhook Specifications

| Setting | Value |
|---------|-------|
| **Protocol** | HTTPS POST |
| **Authentication** | Shared secret in `X-Finnhub-Secret` header |
| **Signature Algorithm** | SHA256 (binary) converted to hex |
| **Timeout** | Server expects 2xx response within 30 seconds |
| **Retry Policy** | Finnhub retries on non-2xx responses |
| **Disable Threshold** | Consecutive day of failures |
| **Max Payload Size** | 1 MB (typically < 100 KB per event) |

## Security Best Practices

1. ✅ **Verify Secret** - Always validate the `X-Finnhub-Secret` header against your configured shared secret
2. ✅ **Use HTTPS** - Configure SSL/TLS on production
3. ✅ **Rotate Secret** - Change webhook secret periodically
4. ✅ **Monitor Logs** - Set up alerts for signature failures
5. ✅ **Rate Limiting** - Implement rate limiting on webhook endpoint
6. ✅ **Error Handling** - Always return 2xx to acknowledge
7. ✅ **Async Processing** - Don't block webhook handler with heavy logic

## Troubleshooting

### Webhook Not Receiving Events

1. **Check Webhook URL**
   - Verify domain is publicly accessible
   - Test with: `curl -I https://your-domain.com:8000/webhooks/finnhub/health`

2. **Check Firewall**
   - Ensure port 8000 is open to Finnhub IP ranges
   - Ask Finnhub for their IP ranges if needed

3. **Check Configuration**
   - Verify webhook is enabled in Finnhub dashboard
   - Check event type filters are set correctly
   - Verify no symbol filters are excluding your stocks

4. **Check Logs**
   - Look for signature verification failures
   - Check for connection errors

### Signature Verification Failing

1. **Verify Secret**
   - Confirm `FINNHUB_WEBHOOK_SECRET` in .env matches Finnhub dashboard
   - Default: `d7jkg4pr01qhf13f7jng`

2. **Test with Sample Event**
   - Use Finnhub dashboard "Send Test Event" button
   - Check logs for exact error

### Events Stopped After Disabling

1. **Check Webhook Status**
   - Go to Finnhub dashboard → Webhooks
   - Look for "disabled" status or error messages

2. **Reason Usually:**
   - Server returned non-2xx responses
   - Network connectivity issues
   - Signature mismatch

3. **Recovery:**
   - Fix the underlying issue
   - Delete and re-add the webhook
   - Test with sample event

## Advanced: Custom Event Handlers

To add support for new event types:

1. **Edit** `app/api/routes_webhooks.py`
2. **Add handler function:**
   ```python
   async def _process_custom_event(payload: dict) -> None:
       """Process custom webhook event."""
       try:
           # Your logic here
           logger.info("custom_event_processed", extra={...})
       except Exception as e:
           logger.exception(f"Error processing custom event: {e}")
   ```
3. **Register in `handle_finnhub_webhook()`:**
   ```python
   elif event_type == "custom_event":
       await _process_custom_event(payload)
   ```
4. **Test** with curl or sample event

## Support

- **Finnhub Docs:** https://finnhub.io/docs/api/rest-webhooks
- **API Status:** https://status.finnhub.io
- **Support Email:** support@finnhub.io

## References

- [Finnhub Webhook Documentation](https://finnhub.io/docs/api/rest-webhooks)
- [HMAC-SHA256 Signature Verification](https://en.wikipedia.org/wiki/HMAC)
- [JSON Webhook Best Practices](https://www.rfc-editor.org/rfc/rfc6455)
