#!/bin/bash

# Create log file
touch /var/log/pipeline.log

# Every day at 1am
echo "0 1 * * 0 cd /app && export \$(cat .env | xargs) && /usr/local/bin/python3 main.py >> /var/log/pipeline.log 2>&1" | crontab -

# Start cron daemon
service cron start

# Verify cron is running
if ! pgrep cron > /dev/null; then
    echo "ERROR: Failed to start cron daemon"
fi

# Run the pipeline once at startup if no database exists
if [ ! -f "/app/output/latest/db.duckdb" ]; then
    echo "No database found, running initial pipeline..."
    python main.py
fi

# Start FastAPI
uvicorn api:app --host 0.0.0.0 --port 5001