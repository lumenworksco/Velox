# Runbook: Database Corruption

## Symptoms
- Log message: `database disk image is malformed`
- Log message: `PROD-005: Migration failed`
- SQLite errors: `OperationalError: database is locked` (persistent, not transient)
- Bot crashes on startup during `init_db()`
- Queries return unexpected empty results

## Diagnosis Steps

1. **Check database integrity:**
   ```bash
   sqlite3 bot.db "PRAGMA integrity_check;"
   ```

2. **Check WAL status:**
   ```bash
   sqlite3 bot.db "PRAGMA journal_mode;"
   ls -la bot.db bot.db-wal bot.db-shm
   ```

3. **Check schema version:**
   ```bash
   sqlite3 bot.db "SELECT * FROM schema_version ORDER BY version;"
   ```

4. **Check for lock files:**
   ```bash
   fuser bot.db 2>/dev/null  # Linux
   lsof bot.db 2>/dev/null   # macOS
   ```

5. **Check disk space:**
   ```bash
   df -h .
   ```

## Resolution Steps

### Scenario 1: Transient Lock
1. Stop the bot process
2. Wait 5 seconds for WAL checkpoint
3. Restart the bot

### Scenario 2: WAL Corruption
1. Stop the bot
2. Force WAL checkpoint:
   ```bash
   sqlite3 bot.db "PRAGMA wal_checkpoint(TRUNCATE);"
   ```
3. If that fails, copy the database (WAL is merged on copy):
   ```bash
   sqlite3 bot.db ".backup bot_recovered.db"
   mv bot.db bot_corrupted.bak
   mv bot_recovered.db bot.db
   ```

### Scenario 3: Full Corruption
1. Stop the bot
2. Check for disaster recovery backup:
   ```bash
   ls -lt backups/bot.db.* 2>/dev/null || ls -lt *.bak 2>/dev/null
   ```
3. Restore from backup:
   ```bash
   cp backups/bot.db.latest bot.db
   ```
4. If no backup exists, recreate:
   ```bash
   mv bot.db bot.db.corrupted
   python3 -c "import database; database.init_db(); database.run_migrations()"
   ```
   Note: Trade history and analytics will be lost. Open positions should be
   reconciled from broker state on next startup.

### Scenario 4: Schema Migration Failure
1. Check which migration failed:
   ```bash
   sqlite3 bot.db "SELECT * FROM schema_version ORDER BY version DESC LIMIT 5;"
   ```
2. Check bot.log for the specific error
3. If the migration was partially applied, you may need to manually fix the schema
4. After fixing, update schema_version manually or re-run the bot

## Prevention
- Enable `PRAGMA synchronous=FULL` (already set in ConnectionPool)
- Ensure WAL mode is active (`PRAGMA journal_mode=WAL`)
- Set up periodic database backups (disaster recovery module)
- Monitor disk space usage
- Never kill -9 the bot during market hours if possible — use SIGTERM for graceful shutdown
- Increase DB_POOL_SIZE if seeing frequent pool exhaustion (T4-004)
