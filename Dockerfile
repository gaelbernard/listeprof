FROM python:3.11-slim

WORKDIR /app

# Install cron
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole "code" directory into /app
COPY code ./code

# Install dspace-rest-python from inside /app/code
RUN pip install /app/code/dspace-rest-python

# Copy application code
COPY main.py api.py start.sh ./

COPY .env .

# Make start script executable
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]