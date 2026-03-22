# Runbook: Model Degradation

## Symptoms
- Prometheus alert: `ModelStale`
- Log message: `Alpha decay CRITICAL: {strategy} — consider demoting`
- Declining win rate or Sharpe ratio over recent days
- ML signal scorer producing consistently low confidence scores
- Walk-forward validation showing out-of-sample performance drop

## Diagnosis Steps

1. **Check model training history:**
   ```bash
   sqlite3 bot.db "SELECT timestamp, strategy, test_f1, test_precision, train_samples FROM model_performance ORDER BY timestamp DESC LIMIT 10;"
   ```

2. **Check strategy performance by day:**
   ```bash
   sqlite3 bot.db "SELECT strategy, DATE(exit_time) as day, COUNT(*) as trades, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, ROUND(AVG(pnl_pct)*100, 2) as avg_pnl_pct FROM trades WHERE exit_time >= date('now', '-7 days') GROUP BY strategy, day ORDER BY day DESC, strategy;"
   ```

3. **Check alpha decay monitor report:**
   ```bash
   grep "alpha decay" bot.log | tail -10
   ```

4. **Check if market regime has shifted:**
   ```bash
   grep "regime" bot.log | tail -20
   ```

5. **Check feature importance drift:**
   ```bash
   sqlite3 bot.db "SELECT features_used FROM model_performance ORDER BY timestamp DESC LIMIT 1;"
   ```

## Resolution Steps

### Step 1: Assess Severity
- **Warning** (Sharpe declining but positive): Monitor for 2-3 more days
- **Critical** (Sharpe negative or win rate < 40%): Take action immediately

### Step 2: Retrain Models
```bash
python3 main.py --walkforward
```
This runs walk-forward validation and retrains models on recent data.

### Step 3: Check for Regime Mismatch
- If the market regime has shifted (e.g., from trending to mean-reverting),
  the model may need regime-specific features or retraining
- Review HMM regime detector output
- Consider temporarily reducing allocation to the degraded strategy

### Step 4: Reduce Allocation
If retraining does not improve performance:
1. Edit `config.py` STRATEGY_ALLOCATIONS
2. Reduce the degraded strategy's weight
3. Redistribute to better-performing strategies

### Step 5: Emergency Demotion
For persistent degradation:
1. Set the strategy allocation to 0 in config
2. Or add it to a "shadow mode" list for paper-only evaluation
3. Monitor shadow trades to detect recovery

## Prevention
- Run walk-forward validation weekly (Sunday task is auto-scheduled)
- Monitor alpha decay metrics daily at EOD
- Set up the `ModelStale` Prometheus alert (fires if model not retrained in 7 days)
- Use the adaptive allocator to automatically reduce allocation to underperforming strategies
- Maintain a diverse strategy set to hedge against single-strategy degradation
