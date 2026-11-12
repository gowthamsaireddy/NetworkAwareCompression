#!/usr/bin/env bash
# ==============================================================================
# Network-Aware Compression - Ubuntu Sender Launch Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo " Starting Network-Aware Compression Sender Node (Ubuntu)"
echo "========================================================================"

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment (./venv)..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "[INFO] Activating virtual environment (../venv)..."
    source ../venv/bin/activate
else
    echo "[WARN] No virtual environment found at ./venv. Using system python3."
fi

# Verify Python
PYTHON_BIN="$(which python3 || which python)"
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3 is not found in PATH."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON_BIN --version) at $PYTHON_BIN"

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "[WARN] config.json not found in root. Using default fallback configuration."
fi

# Launch application
exec "$PYTHON_BIN" main_sender.py "$@"

