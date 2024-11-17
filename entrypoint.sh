#!/bin/bash

# Enable error handling and debug output
set -e
set -x

echo "Starting Xvfb..."
# Start Xvfb
Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=$!

echo "Waiting for Xvfb to start..."
sleep 2

# Verify Xvfb is running
if ! ps -p $XVFB_PID > /dev/null; then
    echo "Error: Xvfb failed to start"
    exit 1
fi

echo "Testing Chrome..."
# Test Chrome installation without running it
echo "Checking Chromium location..."
which chromium || echo "Chromium not in PATH"
ls -l /usr/bin/chromium* || echo "No Chromium binaries found"

echo "Checking ChromeDriver location..."
which chromedriver || echo "ChromeDriver not in PATH"
ls -l /usr/bin/chromedriver* || echo "No ChromeDriver binaries found"

echo "Checking Chrome directories..."
ls -l /var/lib/chrome || echo "Chrome directory not found"

# Trap SIGTERM and SIGINT
cleanup() {
    echo "Cleaning up..."
    if [ -n "$XVFB_PID" ]; then
        kill $XVFB_PID || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "Starting Streamlit..."
# Start Streamlit with specific options
exec streamlit run \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.serverAddress=0.0.0.0 \
    --server.enableCORS=false \
    main.py
