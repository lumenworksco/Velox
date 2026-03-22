# Runbook: PDT Violation Risk

## Symptoms
- Log message: `PDT: Day trade count approaching limit`
- Log message: `PDT compliance: trade blocked`
- Bot skipping signals with reason `pdt_limit`
- Account equity below $25,000 and multiple day trades in the rolling 5-day window

## Diagnosis Steps

1. **Check current day trade count:**
   ```bash
   sqlite3 bot.db "SELECT COUNT(*) FROM trades WHERE DATE(entry_time) >= DATE('now', '-5 days') AND DATE(entry_time) = DATE(exit_time);"
   ```

2. **Check account equity (PDT threshold is $25,000):**
   ```bash
   curl http://localhost:8080/health | python3 -m json.tool
   ```

3. **Check PDT tracker state:**
   ```bash
   grep -i "pdt" bot.log | tail -20
   ```

4. **List recent day trades (entered and exited same day):**
   ```bash
   sqlite3 bot.db "SELECT symbol, strategy, entry_time, exit_time, pnl FROM trades WHERE DATE(entry_time) >= DATE('now', '-5 days') AND DATE(entry_time) = DATE(exit_time) ORDER BY entry_time DESC;"
   ```

5. **Check if account is flagged as pattern day trader:**
   - Log into Alpaca dashboard
   - Check account status for PDT flag

## Resolution Steps

### If Under $25K Equity and Approaching 3 Day Trades

1. **Do NOT make additional day trades.** The bot's PDT compliance module
   should automatically block new day trades when approaching the limit.

2. **Convert remaining positions to overnight holds:**
   - The bot automatically designates positions as overnight holds when the
     PDT limit is near

3. **Wait for the 5-day rolling window to clear:**
   - Day trades older than 5 business days fall off the count
   - Check which trades will roll off:
     ```bash
     sqlite3 bot.db "SELECT entry_time, symbol FROM trades WHERE DATE(entry_time) >= DATE('now', '-6 days') AND DATE(entry_time) = DATE(exit_time) ORDER BY entry_time;"
     ```

### If Already Flagged as PDT

1. **Deposit funds to bring equity above $25,000** (removes PDT restriction)
2. **Or request a one-time PDT reset** from Alpaca (limited to once per account)
3. **Or switch to a cash account** (no PDT rule, but no margin and T+2 settlement)

### Emergency: Bot Making Trades Despite PDT Limit

1. **Activate kill switch:**
   ```bash
   curl -X POST http://localhost:8080/api/kill-switch -H "Content-Type: application/json" -d '{"reason": "pdt_emergency"}'
   ```

2. **Stop the bot:**
   ```bash
   kill -TERM $(pgrep -f "main.py")
   ```

3. **Review PDT tracker code for bugs**

## Prevention
- Keep account equity above $25,000 at all times
- The PDT compliance module (`compliance/pdt.py`) automatically tracks day trade count
- Configure `PDT_MAX_DAY_TRADES=3` (default) in config — the bot reserves 1 trade as buffer
- Use overnight hold strategies (PEAD, OvernightManager) to reduce day trade frequency
- Monitor the rolling 5-day day trade count in the dashboard
- Set up alerts for account equity approaching $25K threshold
