#!/bin/bash

# Automated backup script for Maternal Database
# This script should be run inside the backup container

set -e

# Configuration
DB_NAME=${DB_NAME:-maternal_db}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-postgres}
DB_HOST=${DB_HOST:-db}
DB_PORT=${DB_PORT:-5432}
BACKUP_DIR="/backups"
RETENTION_DAYS=30

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/maternal_backup_$TIMESTAMP.sql"
COMPRESSED_BACKUP="$BACKUP_FILE.gz"

echo "[$(date)] Starting backup process..."

# Create backup
echo "[$(date)] Creating backup: $BACKUP_FILE"
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "[$(date)] Backup created successfully"
    
    # Compress backup
    echo "[$(date)] Compressing backup..."
    gzip "$BACKUP_FILE"
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] Backup compressed successfully: $COMPRESSED_BACKUP"
        echo "[$(date)] Backup size: $(du -h "$COMPRESSED_BACKUP" | cut -f1)"
    else
        echo "[$(date)] ERROR: Failed to compress backup"
        exit 1
    fi
else
    echo "[$(date)] ERROR: Backup failed"
    exit 1
fi

# Clean up old backups
echo "[$(date)] Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "maternal_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

echo "[$(date)] Backup process completed successfully"
