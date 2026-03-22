# Runbook: Orphaned Position

## Symptoms
- Position exists at broker (Alpaca) but not in bot's database
- Position exists in database but not at broker
- Reconciliation log: `Position mismatch detected`
- Dashboard shows position count different from broker count
- Unexpected P&L discrepancies

## Diagnosis Steps

1. **Check bot's tracked positions:**
   ```bash
   sqlite3 bot.db "SELECT symbol, strategy, side, qty, entry_price, entry_time FROM open_positions;"
   ```

2. **Check broker positions (via API):**
   ```bash
   curl http://localhost:8080/api/positions | python3 -m json.tool
   ```

3. **Compare the two lists manually for mismatches.**

4. **Check reconciliation logs:**
   ```bash
   grep -i "reconcil\|mismatch\|orphan" bot.log | tail -20
   ```

5. **Check for recently cancelled/filled orders that may have caused the drift:**
   ```bash
   sqlite3 bot.db "SELECT * FROM execution_analytics WHERE submitted_at >= datetime('now', '-1 hour') ORDER BY submitted_at DESC;"
   ```

## Resolution Steps

### Scenario 1: Position at Broker, Missing from Bot DB

This typically happens after a crash or if the bot was restarted while orders were pending.

1. **Option A — Let sync handle it:**
   The `sync_positions_with_broker()` function runs every scan cycle and should
   detect and adopt orphaned broker positions. Wait one scan cycle.

2. **Option B — Manual adoption:**
   ```bash
   sqlite3 bot.db "INSERT INTO open_positions (symbol, strategy, side, entry_price, qty, entry_time, hold_type) VALUES ('SYMBOL', 'MANUAL', 'buy', PRICE, QTY, datetime('now'), 'day');"
   ```

3. **Option C — Close the orphaned position at broker:**
   - Log into Alpaca dashboard
   - Close the position manually
   - The bot will not track this trade's P&L

### Scenario 2: Position in Bot DB, Missing from Broker

This can happen if a stop/limit order filled at the broker but the bot didn't
receive the fill notification (e.g., WebSocket disconnect).

1. **Check if the position was closed at broker:**
   ```bash
   grep "SYMBOL" bot.log | grep -i "fill\|close\|exit" | tail -10
   ```

2. **Remove the stale position from DB:**
   ```bash
   sqlite3 bot.db "DELETE FROM open_positions WHERE symbol = 'SYMBOL';"
   ```

3. **Log the trade manually if fill info is available:**
   Check Alpaca order history for the actual fill price and time.

### Scenario 3: Quantity Mismatch

1. Check if partial fills or partial exits occurred
2. Update the quantity in the database to match broker:
   ```bash
   sqlite3 bot.db "UPDATE open_positions SET qty = NEW_QTY WHERE symbol = 'SYMBOL';"
   ```

## Prevention
- Ensure WebSocket monitoring is active (`WEBSOCKET_MONITORING=true`)
- The reconciliation module runs every 30 minutes by default (RECONCILIATION_INTERVAL)
- Use `sync_positions_with_broker()` which runs every scan cycle
- Avoid `kill -9` — always use SIGTERM for graceful shutdown
- After any crash, verify positions match before the next trading session
- Review reconciliation alerts if they fire frequently
