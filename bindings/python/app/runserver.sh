#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

gunicorn --bind 0.0.0.0:5000 wsgi:app \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --log-level info \
    --access-logfile logs/access.log \
    --error-logfile logs/app.log \
    --capture-output \
    --enable-stdio-inheritance