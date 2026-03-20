#!/bin/bash
# Restore bot.db from a backup
#
# Usage: bash scripts/restore.sh [backup_file.db.gz]
# If no file specified, lists available backups.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOT_DIR="$(dirname "$SCRIPT_DIR")"
DB_FILE="$BOT_DIR/bot.db"
BACKUP_DIR="$BOT_DIR/backups"

if [ -z "$1" ]; then
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/bot_*.db.gz 2>/dev/null || echo "  (none)"
    echo
    echo "Usage: bash scripts/restore.sh backups/bot_YYYYMMDD_HHMMSS.db.gz"
    exit 0
fi

BACKUP_FILE="$1"
if [ ! -f "$BACKUP_FILE" ]; then
    # Try relative to BOT_DIR
    BACKUP_FILE="$BOT_DIR/$1"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $1"
    exit 1
fi

# Safety: back up current DB first
if [ -f "$DB_FILE" ]; then
    cp "$DB_FILE" "${DB_FILE}.pre_restore"
    echo "Current DB backed up to ${DB_FILE}.pre_restore"
fi

# Decompress and restore
TEMP_FILE="/tmp/velox_restore_$$.db"
gunzip -c "$BACKUP_FILE" > "$TEMP_FILE"

# Verify
TABLES=$(sqlite3 "$TEMP_FILE" "SELECT count(*) FROM sqlite_master WHERE type='table'" 2>/dev/null)
if [ -z "$TABLES" ] || [ "$TABLES" -eq 0 ]; then
    echo "ERROR: Backup appears corrupt (no tables found)"
    rm -f "$TEMP_FILE"
    exit 1
fi

mv "$TEMP_FILE" "$DB_FILE"
echo "Restored from $BACKUP_FILE ($TABLES tables)"
