#!/bin/bash
set -e

# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "Starting DeepSeek OCR service..."

# You can start just the server:
# python3 serve_pdf.py

# Or you can start it using the watchdog to handle auto-restarts and OOM:
python3 watchdog_restart.py "$@"
