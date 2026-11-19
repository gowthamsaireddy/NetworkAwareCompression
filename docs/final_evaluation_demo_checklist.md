# Two-Laptop Live Deployment Demonstration Checklist & Troubleshooting Guide

**Project**: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation  
**Deployment Nodes**:
- **Sender Node**: Ubuntu Linux (Laptop A / Edge Field Acquisition)
- **Receiver Node**: Windows 11 (Laptop B / Hospital PACS & Diagnostic Server)
- **Physical Link**: Direct Cat6 / Cat5e Ethernet Cable (or Shared Isolated Clinical Subnet)

---

## 1. Pre-Demonstration Physical Setup (T minus 15 Minutes)

### Step 1: Physical Link & IP Addressing
1. Connect Ethernet cable directly between Ubuntu Laptop and Windows Laptop.
2. **Configure Windows Receiver Static IP**:
   * Open *Settings $\rightarrow$ Network & Internet $\rightarrow$ Ethernet $\rightarrow$ IP assignment: Edit (Manual)*.
   * IPv4 Address: `172.30.11.108` (or your configured subnet IP).
   * Subnet Mask: `255.255.255.0` (Prefix: `24`).
   * Gateway: Leave blank.
3. **Configure Ubuntu Sender Static IP**:
   * Open terminal:
     ```bash
     sudo ip addr add 172.30.11.107/24 dev eth0
     sudo ip link set dev eth0 up
     ```
4. **Verify Physical Connectivity**:
   * On Ubuntu: `ping -c 3 172.30.11.108`
   * Confirm round-trip latency $\le 1.0\text{ ms}$ with $0\%$ packet loss.

### Step 2: Windows Firewall Port Permission
1. On Windows, ensure inbound TCP port 5000 is open:
   ```powershell
   netsh advfirewall firewall add rule name="NAC_Receiver_Port_5000" dir=in action=allow protocol=TCP localport=5000
   ```

---

## 2. Live Demonstration Execution Script (Step-by-Step)

### Phase 1: Launch Receiver Node (Windows 11)
1. Open PowerShell / Command Prompt in the repository root:
   ```powershell
   .\venv\Scripts\python.exe main_receiver.py
   ```
2. The Receiver GUI launches with the dark clinical theme.
3. Navigate to **TCP Receiver** tab:
   * Verify Listen IP is `0.0.0.0` and Port is `5000`.
   * Click **Start Listening Server**.
   * Verify card status updates to: `Listening on 5000` (green).
4. Navigate to **Model Management** tab:
   * Confirm pre-loaded checkpoint `best_unet.pth` on device `CPU` (or `CUDA` if NVIDIA GPU is present). Status must show: `Loaded (Frozen)`.

---

### Phase 2: Launch Sender Node (Ubuntu)
1. Open terminal in the repository root:
   ```bash
   source venv/bin/activate
   python3 main_sender.py
   ```
2. The Sender GUI launches.
3. Navigate to **TCP Transmission** tab:
   * Enter Receiver IP: `172.30.11.108`, Port: `5000`.
   * Click **Probe Receiver**. Confirm popup: `Receiver node reachable on port 5000`.
   * Click **Connect to Receiver**. Status card updates to: `Connected`.

---

### Phase 3: Demonstration of Option A (Normal JPEG Compression)
1. On Sender, navigate to **Compression & Quality** tab.
2. Select **Option A: Normal Compression (Standard Uniform JPEG)**.
3. Click **Select Image File** $\rightarrow$ Choose `data/BUSI/benign/benign (104).png`.
4. Select quality radio button **Q70 (Balanced)**.
5. Click **Compress Current Image**:
   * Original Preview shows uncompressed frame ($777 \times 578\text{ px}$, $456\text{ KB}$).
   * Compressed Preview updates letterboxed.
   * Metric Cards display: Size Reduction $\approx 89.7\%$, Compression Ratio $\approx 9.74:1$, PSNR $\approx 42.9\text{ dB}$, SSIM $\approx 0.984$.
6. Navigate to **TCP Transmission** tab:
   * Active Payload Banner displays: `Active Payload: Normal JPEG | Image: benign/benign (104) | Q: 70 | Payload: 47,139 B`.
   * Click **Send Active Image Frame**.
   * Progress bar fills smoothly. Telemetry updates:
     * `Measured Wire Time: ~0.0034 s`
     * `Measured Wire Throughput: ~110,915 Kbps`
     * `Receiver Checksum ACK: Verified (ACK OK)`
     * `Round-Trip Latency (RTT): ~0.012 s`
   * Transfer history table adds a new row with green `ACK OK` badge.

---

### Phase 4: Receiver Inspection & Lesion Segmentation Inference
1. Switch to Windows Receiver screen.
2. Check **TCP Receiver** tab: Real-time event log shows verified receipt of `benign/benign (104)` with SHA-256 match.
3. Navigate to **Image Viewer** tab: Click `benign/benign (104)` and click **Send to Segmentation Pipeline**.
4. Navigate to **Segmentation** tab:
   * Left Card (**Received Image**): Displays the complete $777 \times 578$ ultrasound image letterboxed without clipping or zooming.
   * Click **Run Segmentation on Selected Image**:
     * Middle Card (**Predicted Lesion Mask**): Shows $256 \times 256$ binary mask isolating the lesion.
     * Right Card (**Lesion Boundary Overlay**): Displays the emerald-green contour precisely delineating the tumor border directly on the full-resolution ultrasound frame.
     * Metric cards display: `Inference Latency: ~79 ms`, `Processing Resolution: 256 × 256`, `Lesion Area: 442 px`.
5. Navigate to **Evaluation (Dice/IoU)** tab:
   * System automatically matches ground-truth mask `data/BUSI/benign/benign (104)_mask.png`.
   * Metric cards display: **Dice: 0.8752**, **IoU: 0.7782**, **PSNR: 43.29 dB**, **SSIM: 0.9841**.

---

### Phase 5: Demonstration of Option B (ML-Based Region-Adaptive Compression)
1. Switch to Ubuntu Sender screen $\rightarrow$ Navigate to **Compression & Quality** tab.
2. Select **Option B: ML-Based Region-Adaptive Compression**.
3. Configure parameters:
   * Lesion Margin: `15 px`
   * ROI Quality: `90`
   * Background Quality: `20`
   * ROI Locator: `ML Model (roi_detector_v1)`
4. Check checkbox: `Show Preserved ROI Boundary Contour`.
5. Click **Compress Current Image**:
   * Right Preview displays the region-adaptive compressed output with blue contour demarcating the preserved ROI zone.
   * Preserved ROI PSNR shows **49.70 dB** (near-lossless fidelity inside the tumor!).
   * Full-image size drops to **48,481 B** ($89.4\%$ reduction).
6. Navigate to **TCP Transmission** tab $\rightarrow$ Click **Send Active Image Frame**.
7. Switch to Windows Receiver $\rightarrow$ Run Segmentation & Evaluation:
   * Dice score achieves **0.8664** and IoU **0.7643**—identical clinical diagnostic accuracy to uncompressed Q90, while transmitting $37.9\%$ fewer bytes than uniform Q90!

---

### Phase 6: Demonstration of Network Emulation (tc/netem)
1. On Ubuntu Sender, navigate to **Network Emulation** tab.
2. Select Profile: **3G Rural Telemedicine (384 kbps, 150ms delay, 1% loss)**.
3. Click **Apply Emulation Profile**:
   * Terminal prompt asks for confirmation.
   * Kernel qdisc rule applies: `tc qdisc replace dev eth0 root netem delay 150ms loss 1% rate 384kbit`.
4. Transmit an image frame:
   * Observe Measured Wire Time increases to $\approx 1.05\text{ s}$.
   * Measured throughput reflects the 384 Kbps ceiling.
   * SHA-256 integrity continues to pass with $100\%$ verification because TCP automatically retransmits any lost packets!
5. Click **Reset / Restore Baseline Network**: Confirms qdisc rule removal.

---

### Phase 7: Persistent Research Database & Audit Verification
1. On Windows Receiver, navigate to **Experiment Results** tab:
   * Click **Refresh Records**.
   * Observe all 28+ logged trials in the interactive table.
   * Confirm columns `Tx Time (s)` and `Throughput (Kbps)` show genuine physical measurements (e.g. `0.0033 s`, `119,800.00 Kbps`).
   * Filter by Quality (`Q90`, `Q70`, etc.) or Mode (`Fixed Rate` vs `Adaptive`).
   * Click **Export CSV** $\rightarrow$ Saves `experiments/experiments.csv`.
   * Click **Export Summary Report** $\rightarrow$ Compiles complete Markdown research audit report.

---

## 3. Real-Time Troubleshooting & Fault-Recovery Playbook

| Symptom / Error | Root Cause | Immediate Remediation Step |
| :--- | :--- | :--- |
| **"Connection Refused" on Sender Probe** | Windows Receiver daemon not running or blocked by firewall | 1. Confirm `Start Listening Server` was clicked on Windows.<br>2. In Windows PowerShell: `netstat -ano \| findstr 5000` to confirm listening.<br>3. Temporarily disable Windows Public Firewall or run the `netsh` rule above. |
| **"Socket Timeout" during Send** | Network interface misconfiguration or cable unplugged | 1. Verify physical link lights on Ethernet ports.<br>2. Run `ping 172.30.11.108` from Ubuntu terminal.<br>3. Check if sender IP is on the same subnet (`172.30.11.x`). |
| **"Checksum Mismatch" Alert on Receiver** | Payload corrupted in-flight | 1. Review sender logs to check if bit-flip injection test was active.<br>2. Retransmit frame; TCP automatic retransmission will re-send clean stream. |
| **Image Appears Squashed / Collapsed into Slit** | Screen resolution $<768\text{p}$ compressed layout | 1. Ensure latest code with `ResizableImageLabel` and `QScrollArea` is pulled.<br>2. Resize window or use mouse wheel to scroll down smoothly. |
| **"No Model Loaded" Warning on Segmentation** | Checkpoint unmounted | 1. Open Receiver **Model Management** tab.<br>2. Click **Browse Checkpoint** $\rightarrow$ Select `models/best_unet.pth`.<br>3. Click **Mount Model**. Status must display green `Loaded (Frozen)`. |
| **"No Ground Truth Mask Found" in Evaluation** | Image ID naming mismatch | 1. Verify BUSI dataset directory structure: `data/BUSI/benign/` must contain matching `benign (X)_mask.png`.<br>2. If normal case selected, system correctly reports `Normal Tissue (Empty Mask, Dice=1.0)`. |
| **Linux `tc` Command Fails with "Operation Not Permitted"** | Non-root execution | On Ubuntu terminal, ensure user has sudo rights, or launch sender script with sudo if running network shaping commands. |
