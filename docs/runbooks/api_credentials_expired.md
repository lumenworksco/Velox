# Runbook: API Credentials Expired or Invalid

## Symptoms
- Log message: `FATAL: Cannot connect to Alpaca API`
- Log message: `Failed to fetch bars for {symbol}: 403 Forbidden`
- Prometheus alert: `APIFailureRate` or `APIFailureRateCritical`
- Bot exits on startup with credential error
- Readiness probe returns 503 with broker check failing

## Diagnosis Steps

1. **Check if the bot process is running:**
   ```bash
   ps aux | grep main.py
   ```

2. **Check recent API errors in logs:**
   ```bash
   grep -i "401\|403\|forbidden\|unauthorized\|credential" bot.log | tail -20
   ```

3. **Verify environment variables are set:**
   ```bash
   echo "API_KEY set: $([ -n "$ALPACA_API_KEY" ] && echo yes || echo NO)"
   echo "API_SECRET set: $([ -n "$ALPACA_API_SECRET" ] && echo yes || echo NO)"
   ```

4. **Test API connectivity manually:**
   ```bash
   curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
        https://paper-api.alpaca.markets/v2/account | python3 -m json.tool
   ```

5. **Check if keys are for paper vs live:**
   - Paper keys start with `PK` (key) and `PS` (secret) — NOT always
   - Live keys use the live endpoint: `api.alpaca.markets`
   - Check `ALPACA_LIVE` environment variable

## Resolution Steps

1. **Regenerate API keys:**
   - Log into [Alpaca Dashboard](https://app.alpaca.markets)
   - Go to API Keys section
   - Generate new key pair
   - Update environment variables or `.env` file

2. **Update environment:**
   ```bash
   export ALPACA_API_KEY="new_key_here"
   export ALPACA_API_SECRET="new_secret_here"
   ```

3. **Restart the bot:**
   ```bash
   python3 main.py
   ```

4. **Verify connectivity:**
   ```bash
   curl http://localhost:8080/ready | python3 -m json.tool
   ```

## Prevention
- Set up key rotation reminders (Alpaca keys do not expire but can be revoked)
- Use a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault) instead of env vars
- Monitor the `APIFailureRate` Prometheus alert to catch issues early
- Keep backup API keys in a secure location
