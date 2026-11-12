#!/bin/bash
# ==============================================================================
# Network-Aware Adaptive Compression - Ubuntu Sender Deployment Setup
# ==============================================================================
set -e

echo "=== [1/6] Validating System & Python Prerequisites ==="
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found. Please install Python 3.10+ (sudo apt-get install python3 python3-venv python3-pip)"
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[+] Detected Python version: $PY_VER"

echo "=== [2/6] Checking Linux Traffic Control (tc / netem) ==="
if command -v tc &> /dev/null; then
    echo "[+] Linux tc (iproute2) binary detected: $(which tc)"
else
    echo "[!] 'tc' command not found. Install iproute2: sudo apt-get install -y iproute2"
fi

if modprobe sch_netem 2>/dev/null; then
    echo "[+] Linux sch_netem kernel module available."
else
    echo "[*] Notice: sch_netem could not be loaded directly; if using container/VM, ensure kernel netem is enabled."
fi

echo "=== [3/6] Setting Up Python Virtual Environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[+] Created virtual environment in ./venv"
else
    echo "[+] Reusing existing virtual environment in ./venv"
fi

source venv/bin/activate
pip install --upgrade pip

echo "=== [4/6] Installing Modular Sender Dependencies ==="
if [ -f "requirements/common.txt" ]; then
    pip install -r requirements/common.txt
fi
if [ -f "requirements/sender.txt" ]; then
    pip install -r requirements/sender.txt
fi

echo "=== [5/6] Verifying Runtime Configuration (config.json) ==="
if [ ! -f "config.json" ]; then
    echo "[+] Generating initial default config.json"
    cat <<EOF > config.json
{
  "sender": {
    "receiver_ip": "127.0.0.1",
    "receiver_port": 5000,
    "connection_timeout": 15.0,
    "max_retries": 3,
    "retry_delay_sec": 1.5,
    "dataset_path": "data/BUSI",
    "default_quality": 70,
    "default_network_profile": "unshaped",
    "network_interface": "eth0",
    "log_level": "INFO"
  },
  "receiver": {
    "listen_ip": "0.0.0.0",
    "listen_port": 5000,
    "connection_timeout": 30.0,
    "save_directory": "data/received",
    "model_checkpoint_path": "models/best_unet.pt",
    "device": "auto",
    "auto_process_enabled": true,
    "log_level": "INFO"
  }
}
EOF
else
    echo "[+] Verified existing config.json"
fi

echo "=== [6/6] Permission & Network Setup Guidance ==="
echo "------------------------------------------------------------------"
echo "PRIVILEGE NOTICE FOR NETWORK SHAPING:"
echo "Linux requires elevated privileges to configure traffic shaping via tc/netem."
echo "Option A (Recommended for dev): Run launch script as normal user; simulation fallback handles unshaped links."
echo "Option B (For physical tc shaping): Grant cap_net_admin to tc:"
echo "  sudo setcap cap_net_admin+ep $(which tc)"
echo "------------------------------------------------------------------"
echo "[+] Ubuntu Sender Setup Completed Successfully!"
echo "To launch the Sender application, run: ./run_sender.sh"

