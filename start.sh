#!/bin/bash

# Every Monday at 2am
echo "0 1 * * * cd /app && export \$(cat .env | xargs) && /usr/local/bin/python3 main.py >> /var/log/pipeline.log 2>&1
" | crontab -

# Start cron daemon
cron

# Run the pipeline once at startup if no database exists
if [ ! -f "/app/output/latest/db.duckdb" ]; then
    echo "No database found, running initial pipeline..."
    python main.py
fi

# Start FastAPI
uvicorn api:app --host 0.0.0.0 --port 5001