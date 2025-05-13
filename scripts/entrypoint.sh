#!/bin/bash
set -e

# Create log directory if it doesn't exist
mkdir -p logs

# Download models if they don't exist
python scripts/download_models.py

# Apply database migrations if needed
# python scripts/migrate.py

# Run command
exec "$@"
