# Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

Academic research framework investigating the quantitative impact of network-aware adaptive compression on the transmission of Breast Ultrasound (BUSI) images and downstream deep learning lesion segmentation performance across a physical dual-laptop Ethernet testbed.

---

## 1. Physical Testbed Architecture

```
+------------------------------------------+        Physical Ethernet        +------------------------------------------+
|          LAPTOP 1: SENDER NODE           | <-----------------------------> |         LAPTOP 2: RECEIVER NODE          |
|                                          |                                 |                                          |
| - Host: Windows / Linux                  |                                 | - Windows 11 (Physical Host)             |
| - Guest: Ubuntu VM (Bridged Ethernet)    |                                 | - TCP Receiver Server (Port 5000)        |
| - Dataset Manager (BUSI Dataset)         |                                 | - Binary Frame Assembly & Checksum       |
| - Variable JPEG Engine (Q30-Q90)         |                                 | - Pre-trained Frozen U-Net Inference     |
| - Linux tc / netem Traffic Shaping       |                                 | - Lesion Segmentation (256x256)          |
| - Length-prefixed TCP Client             |                                 | - Ground-truth Matching (Dice / IoU)     |
| - Provisional Adaptive Controller        |                                 | - Experiment Results Logging             |
+------------------------------------------+                                 +------------------------------------------+
```

---

## 2. Repository Structure

```
NetworkAwareCompression/
├── sender/                 # Sender node modules (Ubuntu VM)
│   ├── app.py              # Sender GUI entrypoint
│   ├── dataset_manager.py  # BUSI dataset scanning and indexing
│   ├── compression.py      # JPEG compression engine (Q30, Q50, Q70, Q90)
│   ├── image_metrics.py    # PSNR and SSIM computation
│   ├── network_control.py  # Linux tc / netem traffic shaping
│   ├── tcp_client.py       # TCP client with binary framing
│   └── adaptive_controller.py # Provisional bandwidth-adaptive quality controller
├── receiver/               # Receiver node modules (Windows 11)
│   ├── app.py              # Receiver GUI entrypoint
│   ├── tcp_server.py       # TCP server daemon
│   ├── image_receiver.py   # Image storage and indexing
│   ├── integrity.py        # SHA-256 and framing validation
│   ├── model_manager.py    # Frozen pre-trained U-Net manager
│   ├── segmentation.py     # Preprocessing and segmentation pipeline
│   └── segmentation_metrics.py # Genuine Dice and IoU calculation
├── shared/                 # Common protocol and definitions
│   ├── protocol.py         # Binary framing and socket primitives
│   ├── metadata.py         # Transmission metadata & ACK definitions
│   └── constants.py        # Protocol constants and UI strings
├── common/                 # Common utilities
│   ├── config.py           # Configuration classes and path setup
│   ├── logging_config.py   # Centralized logger setup
│   └── results_manager.py  # SQLite & CSV experiment logger
├── ui/                     # PyQt6 user interfaces
│   ├── sender/             # Sender views (Dashboard, Dataset, Compression, etc.)
│   ├── receiver/           # Receiver views (Dashboard, TCP, Model, Eval, etc.)
│   └── shared/             # Stylesheet, headers, and metric cards
├── models/                 # Model checkpoints directory (never committed to Git)
│   └── README.md
├── data/                   # BUSI dataset root directory (never committed to Git)
│   └── README.md
├── experiments/            # Experiment run logs (SQLite / CSV)
│   └── README.md
├── tests/                  # Unit and integration test suite
│   ├── __init__.py
│   └── test_env.py
├── docs/                   # Architectural documentation
│   └── architecture.md
├── requirements/           # Environment requirement files
│   ├── common.txt          # Shared dependencies
│   ├── sender.txt          # Ubuntu VM dependencies
│   └── receiver.txt        # Windows 11 dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Master project documentation
```

---

## 3. Installation & Setup

### Laptop 1 (Sender — Ubuntu Linux VM)
```bash
# 1. Install system prerequisites (tc/netem and OpenGL dependencies)
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv iproute2 libgl1

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements/sender.txt
```

### Laptop 2 (Receiver — Windows 11)
```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install Python dependencies
pip install -r requirements/receiver.txt
```

---

## 4. Running the Applications

### Sender Application (Ubuntu VM):
```bash
python sender/app.py
```

### Receiver Application (Windows 11):
```powershell
python receiver/app.py
```

---

## 5. Development Phases

- **Phase 1: Environment & Project Foundation** *(Current)*: Project skeleton, dependencies, GUI skeletons, zero-dummy policies.
- **Phase 2: Shared TCP Framing & Communication Protocol**: Socket client/server, partial reads, SHA-256 ACK.
- **Phase 3: Dataset Management & JPEG Compression**: BUSI dataset scanning, Q30-Q90 compression, PSNR/SSIM.
- **Phase 4: Linux Traffic Shaping & Emulation**: `tc` and `netem` profiles (10 Mbps, 2 Mbps, 512 Kbps, 128 Kbps).
- **Phase 5: Deep Learning Segmentation & Evaluation**: External U-Net checkpoint integration, frozen inference, Dice/IoU.
- **Phase 6: Adaptive Compression Integration**: Dynamic channel-adaptive rate selection and end-to-end trials.
- **Phase 7: Empirical Verification & Research Outputs**: Statistical analysis, graphs, publication artifacts.

---

## 6. Strict Research Policies

1. **Zero-Dummy-Data Policy**: No synthetic metrics, mock counts, or fabricated Dice/IoU scores. Uncalculated fields display as `No data available` or `Not calculated`.
2. **Model State**: The U-Net model is strictly loaded in frozen evaluation mode (`torch.no_grad()`). No training occurs inside the receiver.
3. **No Compression of Masks**: Ground-truth masks are never subjected to lossy JPEG compression.

