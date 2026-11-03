# Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

Academic research software framework investigating the quantitative impact of network-aware adaptive compression on the transmission of Breast Ultrasound (BUSI) images and downstream deep learning lesion segmentation performance across a physical dual-laptop Ethernet testbed.

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
| - End-to-End Experiment Orchestrator     |                                 | - Results Visualization & Overlays       |
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
│   ├── adaptive_controller.py # Provisional bandwidth-adaptive quality controller
│   └── experiment_orchestrator.py # End-to-end trial coordination and logging
├── receiver/               # Receiver node modules (Windows 11)
│   ├── app.py              # Receiver GUI entrypoint
│   ├── tcp_server.py       # TCP server daemon
│   ├── image_receiver.py   # Image storage and indexing
│   ├── integrity.py        # SHA-256 and framing validation
│   ├── model_manager.py    # Frozen pre-trained U-Net manager
│   ├── segmentation.py     # Preprocessing and segmentation pipeline
│   └── segmentation_metrics.py # Genuine Dice and IoU calculation
├── shared/                 # Common protocol and definitions
│   ├── protocol.py         # Binary framing and socket primitives (NAC1 / NACK)
│   ├── metadata.py         # Transmission metadata & ACK definitions
│   └── constants.py        # Protocol constants and UI strings
├── common/                 # Common utilities
│   ├── config.py           # Configuration classes and path setup
│   ├── logging_config.py   # Centralized logger setup
│   └── results_manager.py  # SQLite & CSV experiment logger
├── ui/                     # PyQt6 user interfaces
│   ├── sender/             # Sender views (Dashboard, Dataset, Compression, Network, Transmission, Adaptive, Results)
│   ├── receiver/           # Receiver views (Dashboard, Server, Viewer, Model, Segmentation, Evaluation, Results)
│   └── shared/             # Stylesheet, headers, and metric cards
├── models/                 # Model checkpoints directory (never committed to Git)
│   └── README.md
├── data/                   # BUSI dataset root directory (never committed to Git)
│   └── README.md
├── experiments/            # Experiment run logs (SQLite / CSV) & Analysis Tooling
│   ├── analyze_results.py  # Scientific plotting & rate-distortion analysis CLI
│   ├── experiments.sqlite  # SQLite experimental database
│   ├── experiments.csv     # Traceable CSV export
│   └── README.md
├── tests/                  # Automated test suite (41 unit and integration tests)
│   ├── test_env.py
│   ├── test_protocol.py
│   ├── test_dataset_compression.py
│   ├── test_network_telemetry.py
│   ├── test_segmentation_evaluation.py
│   ├── test_adaptive_orchestration.py
│   └── test_analysis.py
├── docs/                   # Architectural & Experimental Documentation
│   ├── architecture.md
│   ├── setup_guide.md
│   ├── protocol_specification.md
│   └── experimental_methodology.md
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

### Receiver Application (Windows 11):
```powershell
python receiver/app.py
```

### Sender Application (Ubuntu VM):
```bash
python sender/app.py
```

### Scientific Results Analysis CLI:
```bash
python experiments/analyze_results.py --db experiments/experiments.sqlite --outdir experiments/plots
```

---

## 5. Development Phases Summary

- **Phase 1: Environment & Project Foundation** *(Completed)*: Repository structure, dependencies, PyQt6 skeletons, Zero-Dummy data enforcement.
- **Phase 2: Shared TCP Framing & Communication Protocol** *(Completed)*: Binary framing (`NAC1` / `NACK`), 64 KB chunked streaming, partial read handling, SHA-256 integrity verification.
- **Phase 3: Dataset Management & JPEG Compression** *(Completed)*: BUSI dataset scanning, multi-mask combination, standardized JPEG compression (Q30, Q50, Q70, Q90), PSNR and SSIM computation.
- **Phase 4: Linux Traffic Shaping & Telemetry** *(Completed)*: Linux `tc`/`netem` kernel qdisc integration, research profiles (10 Mbps, 2 Mbps, 512 Kbps, 128 Kbps), theoretical vs. measured wire time logging.
- **Phase 5: Deep Learning Segmentation & Metrics** *(Completed)*: Pre-trained U-Net checkpoint loading, frozen evaluation mode (`torch.no_grad()`), $256 \times 256$ normalization, mathematical Dice and IoU calculations, color-coded overlays.
- **Phase 6: Provisional Adaptive Quality Controller** *(Completed)*: Provisional rate adaptation logic, EWMA bandwidth estimation ($\alpha=0.3$), fallback priority, and `ExperimentOrchestrator`.
- **Phase 7: Scientific Plotting & Publication Tooling** *(Completed)*: `analyze_results.py` CLI generating publication-grade rate-distortion curves, segmentation degradation curves, latency comparisons, and statistical summary tables.

---

## 6. Strict Research Policies

1. **Zero-Dummy-Data Policy**: No synthetic metrics, mock counts, or fabricated Dice/IoU scores. Uncalculated fields display strictly as `No data available` or `Not calculated`.
2. **Frozen Model State**: The deep learning segmentation model is strictly loaded in frozen evaluation mode (`model.eval()`, `torch.no_grad()`). No training occurs inside the application.
3. **Ground-Truth Protection**: Ground-truth lesion masks are NEVER subjected to lossy JPEG compression.
4. **Wire Delay Traceability**: Theoretical transmission latency ($L_{\text{theo}} = \frac{\text{Bytes} \times 8}{\text{Bandwidth}}$) is explicitly distinguished from measured physical socket transmission latency.

---

## 7. Running the Automated Test Suite

To run all 43 automated unit and integration tests across all modules:
```bash
python -m unittest discover tests
```
