# Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

Academic research software framework investigating the quantitative impact of network-aware adaptive compression on the transmission of Breast Ultrasound (BUSI) images and downstream deep learning lesion segmentation performance across a physical dual-laptop Ethernet testbed.

---

## 1. Physical Testbed Architecture

```
+------------------------------------------+        Physical Ethernet        +------------------------------------------+
|          LAPTOP 1: SENDER NODE           | <-----------------------------> |         LAPTOP 2: RECEIVER NODE          |
|                                          |                                 |                                          |
| - Host: Windows / Linux                  |                                 | - Windows 10/11 (Physical Host)          |
| - Ubuntu VM (Bridged Ethernet)           |                                 | - TCP Receiver Server (Default Port 5000)|
| - Central Config (`config.json`)         |                                 | - Binary Frame Framing & SHA-256 Checksum|
| - Dataset Manager (BUSI Dataset)         |                                 | - Automatic Processing Pipeline          |
| - Variable JPEG Engine (Q30-Q90)         |                                 | - Pre-trained Frozen U-Net Inference     |
| - Linux tc / netem Traffic Shaping       |                                 | - Lesion Segmentation (256x256)          |
| - Dynamic TCP Client with Retry Backoff  |                                 | - Ground-truth Matching (Dice / IoU)     |
| - Provisional Adaptive Controller        |                                 | - Traceable SQLite & CSV Experiment Log  |
| - Automated Batch Experiment Runner      |                                 | - Filtering & Summary Analytics Export   |
| - Telemetry Dashboard & Reachability     |                                 | - Real-time Receiver Monitoring Dashboard|
+------------------------------------------+                                 +------------------------------------------+
```

---

## 2. Repository Structure

```
NetworkAwareCompression/
├── config.json             # Centralized deployment configuration (IP, Port, Timeouts, Model path)
├── main_sender.py          # Sender application top-level entrypoint
├── main_receiver.py        # Receiver application top-level entrypoint
├── setup_ubuntu.sh         # Automated Ubuntu Sender environment provisioning and dependency checks
├── run_sender.sh           # Sender execution wrapper script
├── setup_windows.ps1       # Automated Windows Receiver setup, CUDA verification, and firewall guidance
├── run_receiver.bat        # Windows Receiver launch batch script
├── sender/                 # Sender node modules (Ubuntu VM)
│   ├── app.py              # Sender GUI application controller
│   ├── dataset_manager.py  # BUSI dataset scanning, multi-mask combination, and indexing
│   ├── compression.py      # JPEG compression engine (Q30, Q50, Q70, Q90)
│   ├── image_metrics.py    # PSNR and SSIM computation
│   ├── network_control.py  # Linux tc / netem kernel traffic shaping
│   ├── tcp_client.py       # TCP client with binary framing, reachability probe, and retry backoff
│   ├── adaptive_controller.py # Bandwidth-adaptive quality decision engine
│   └── experiment_orchestrator.py # Automated batch experiment orchestration and atomic ID generation
├── receiver/               # Receiver node modules (Windows 11)
│   ├── app.py              # Receiver GUI application controller
│   ├── tcp_server.py       # Multi-threaded TCP receiver server daemon
│   ├── image_receiver.py   # Path-sanitized image storage with disk safety bounds
│   ├── integrity.py        # SHA-256 checksum and protocol framing validation
│   ├── model_manager.py    # Frozen pre-trained U-Net manager (supports GPU/CPU and device selection)
│   ├── segmentation.py     # 256x256 preprocessing and inference pipeline
│   └── segmentation_metrics.py # Mathematical Dice and IoU calculation with empty mask policies
├── shared/                 # Common protocol and message definitions
│   ├── protocol.py         # Binary framing (NAC1 / NACK), length-prefixed streaming, DoS size bounds
│   ├── metadata.py         # Transmission metadata & ACK dataclass definitions
│   └── constants.py        # Protocol constants and UI strings
├── common/                 # Core framework utilities
│   ├── config.py           # Configuration manager (JSON loading + ENV var overrides)
│   ├── logging_config.py   # Centralized logger setup
│   └── results_manager.py  # Traceable SQLite database & CSV logging manager
├── ui/                     # Modern PyQt6 user interfaces
│   ├── sender/             # Sender views (Dashboard, Dataset, Compression, Network, Transmission, Adaptive, Results)
│   ├── receiver/           # Receiver views (Dashboard, Server, Viewer, Model, Segmentation, Evaluation, Results)
│   └── shared/             # Theme stylesheets, section headers, and metric cards
├── models/                 # Model checkpoints directory (U-Net weights)
│   ├── best_unet.pth       # Trained PyTorch state_dict checkpoint
│   ├── best_unet.pt        # Optimized TorchScript checkpoint
│   └── train_unet.py       # U-Net GPU training script on BUSI splits
├── data/                   # BUSI dataset root directory
│   ├── BUSI/               # Original ultrasound scans and ground-truth masks
│   └── received/           # Verified received compressed images
├── experiments/            # Traceable experimental logs and scientific analysis tooling
│   ├── analyze_results.py  # Publication-grade plotting and rate-distortion analysis CLI
│   ├── experiments.sqlite  # SQLite experimental database
│   ├── experiments.csv     # Traceable raw CSV export
│   └── evaluate_baseline.py# Ground-truth test split baseline evaluation script
├── tests/                  # Comprehensive test suite (55 unit and integration tests)
│   ├── test_env.py
│   ├── test_protocol.py
│   ├── test_dataset_compression.py
│   ├── test_network_and_logging.py
│   ├── test_segmentation_evaluation.py
│   ├── test_adaptive_orchestration.py
│   ├── test_analysis.py
│   └── test_deployment_improvements.py
└── docs/                   # Architectural & Experimental Documentation
    ├── architecture.md
    ├── setup_guide.md
    ├── protocol_specification.md
    └── experimental_methodology.md
```

---

## 3. Quick Start & Deployment

### Sender Node (Ubuntu Linux)

```bash
# 1. Run automated setup script (installs system packages, configures venv, verifies tc/netem)
chmod +x setup_ubuntu.sh run_sender.sh
./setup_ubuntu.sh

# 2. Configure receiver endpoint in config.json (or via env var NAC_RECEIVER_IP)
# Set "receiver_ip": "192.168.10.2" to match your Windows Receiver LAN IP

# 3. Launch Sender GUI
./run_sender.sh
# (or: python3 main_sender.py)
```

### Receiver Node (Windows 10/11)

```powershell
# 1. Run automated setup script in PowerShell
.\setup_windows.ps1

# 2. (One-time) Open Windows Firewall for inbound TCP port 5000:
# Run in Administrator PowerShell:
# New-NetFirewallRule -DisplayName "NetworkAwareCompression-Receiver" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow

# 3. Launch Receiver GUI
.\run_receiver.bat
# (or: python main_receiver.py)
```

---

## 4. Key Production Improvements Implemented

1. **Robust Connection & Retry Management**:
   - Dynamic IP and port configuration via `config.json` or GUI input without application restart.
   - Active RTT reachability probing ("Ping / Probe Receiver") before initiating transmission.
   - Configurable exponential backoff retries (`max_retries=3`, `retry_delay=1.5s`) for handling network fluctuations.
   - Clean socket shutdown (`SHUT_RDWR`) preventing port exhaustion or dangling threads.

2. **DoS Protection & Protocol Hardening**:
   - Strict protocol framing with `NAC1` and `NACK` binary magic numbers.
   - Hardware-protecting payload bounds: metadata capped at 64 KB (`MAX_METADATA_SIZE`), image payloads capped at 50 MB (`MAX_IMAGE_PAYLOAD_SIZE`).
   - Path traversal prevention: sanitizes all incoming `image_id` and `experiment_id` attributes, preventing escape from the storage directory.
   - Pre-write storage safety check: verifies available disk space exceeds image size + 50 MB margin.

3. **Receiver Automatic Processing Pipeline**:
   - Automatically chains: Reception -> SHA-256 Checksum Validation -> Disk Storage -> Frozen U-Net Inference -> Ground-Truth Lookup -> Dice & IoU Calculation -> SQLite & CSV Record Creation -> UI Dashboard Update.
   - Supports both Automatic and Manual segmentation modes via a live toggle on the Receiver Dashboard.

4. **Automated Batch Experimentation**:
   - User-controlled batch runner executing multi-image trials across fixed quality tiers (Q30, Q50, Q70, Q90) or adaptive mode.
   - Dynamic cancellation token allowing non-blocking trial abort with guaranteed restoration of unshaped network state.
   - Atomic timestamped experiment IDs preventing collision or silent overwrites.

5. **Searchable Telemetry & Academic Summary Exports**:
   - Live search input and multi-criteria filters (Quality: Q30-Q90, Mode: Fixed vs Adaptive) across historical trials.
   - One-click "Export Summary Report" generating publication-ready grouped statistical tables ($\text{mean} \pm \text{std}$) for Compression Ratio, PSNR, SSIM, Dice, IoU, Transmission Delay, and Throughput.

---

## 5. Strict Research Policies

1. **Zero-Dummy-Data Policy**: No synthetic metrics, mock counts, or fabricated Dice/IoU scores. Uncalculated fields display strictly as `No data available` or `Not calculated`.
2. **Frozen Model State**: The deep learning segmentation model is strictly loaded in frozen evaluation mode (`model.eval()`, `torch.no_grad()`). No training occurs inside the runtime application.
3. **Ground-Truth Protection**: Ground-truth lesion masks are NEVER subjected to lossy JPEG compression.
4. **Wire Delay Traceability**: Theoretical transmission latency ($L_{\text{theo}} = \frac{\text{Bytes} \times 8}{\text{Bandwidth}}$) is explicitly distinguished from measured physical socket transmission latency.

---

## 6. Running the Automated Test Suite

To run all 55 automated unit and integration tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
All tests run with full isolation and execute in under 3 seconds.
