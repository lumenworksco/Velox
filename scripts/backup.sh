#!/bin/bash
# Automated backup of bot.db with 30-day retention
#
# Usage: bash scripts/backup.sh
# Cron:  0 6 * * * cd /path/to/trading_bot && bash scripts/backup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOT_DIR="$(dirname "$SCRIPT_DIR")"
DB_FILE="$BOT_DIR/bot.db"
BACKUP_DIR="$BOT_DIR/backups"

mkdir -p "$BACKUP_DIR"

if [ ! -f "$DB_FILE" ]; then
    echo "No database found at $DB_FILE"
    exit 0
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/bot_${TIMESTAMP}.db"

# Use SQLite's .backup command for a consistent copy
sqlite3 "$DB_FILE" ".backup '$BACKUP_FILE'"

if [ $? -eq 0 ]; then
    # Verify backup
    TABLES=$(sqlite3 "$BACKUP_FILE" "SELECT count(*) FROM sqlite_master WHERE type='table'")
    echo "Backup OK: $BACKUP_FILE ($TABLES tables)"

    # Compress
    gzip "$BACKUP_FILE"
    echo "Compressed: ${BACKUP_FILE}.gz"
else
    echo "ERROR: Backup failed!"
    exit 1
fi

# Prune backups older than 30 days
find "$BACKUP_DIR" -name "bot_*.db.gz" -mtime +30 -delete
echo "Pruned backups older than 30 days"
