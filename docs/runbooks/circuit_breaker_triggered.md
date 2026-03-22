# Runbook: Circuit Breaker Triggered

## Symptoms
- Log message: `Circuit breaker {YELLOW|RED|BLACK} — skipping scan cycle`
- Prometheus alert: `CircuitBreakerTriggered` or `DailyDrawdownExceeds3Pct`
- Dashboard shows "CB" status instead of "SCANNING"
- No new trades being opened

## Diagnosis Steps

1. **Check current circuit breaker level:**
   ```bash
   curl http://localhost:8080/health | python3 -m json.tool
   ```

2. **Check day P&L:**
   ```bash
   sqlite3 bot.db "SELECT date, day_pnl_pct FROM daily_snapshots ORDER BY date DESC LIMIT 5;"
   ```

3. **Review recent trades for the trigger:**
   ```bash
   sqlite3 bot.db "SELECT symbol, strategy, pnl, pnl_pct, exit_reason, exit_time FROM trades WHERE exit_time >= date('now') ORDER BY exit_time DESC LIMIT 20;"
   ```

4. **Check if a single strategy caused the drawdown:**
   ```bash
   sqlite3 bot.db "SELECT strategy, SUM(pnl) as total_pnl, COUNT(*) as trades FROM trades WHERE exit_time >= date('now') GROUP BY strategy ORDER BY total_pnl;"
   ```

5. **Check VIX / market regime:**
   ```bash
   grep "regime" bot.log | tail -5
   ```

## Resolution Steps

### YELLOW (day P&L -1% to -2%)
- **Action:** No manual intervention needed. Bot continues with reduced position sizes.
- New entries are still allowed but sizing is halved.
- Monitor for further deterioration.

### RED (day P&L -2% to -3%)
- **Action:** Day-hold positions are being auto-closed.
- New entries are blocked until next trading day.
- Verify that positions were actually closed at broker:
  ```bash
  curl http://localhost:8080/api/positions
  ```

### BLACK (day P&L > -3% or manual kill switch)
- **Action:** ALL positions are being liquidated.
- Verify liquidation completed:
  ```bash
  sqlite3 bot.db "SELECT COUNT(*) FROM open_positions;"
  ```
- If positions remain stuck, manually close via Alpaca dashboard.
- Bot will not trade again until manually restarted.

### Manual Reset (after BLACK)
1. Verify all positions are closed at broker
2. Check for orphaned orders: `curl http://localhost:8080/api/orders`
3. Restart the bot: `python3 main.py`
4. Circuit breaker resets on new trading day

## Prevention
- Review strategy allocations if one strategy consistently triggers the breaker
- Consider tightening per-trade stop losses
- Review VIX scaling parameters if breaker triggers during high-vol regimes
- Ensure the vol scalar is properly de-risking during elevated VIX
