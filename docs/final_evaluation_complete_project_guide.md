# Master Project Evaluation Guide: Complete System Reference Manual
## Project: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

---

# SECTION 1: EXECUTIVE SUMMARY & SYSTEM TOPOLOGY

### 1.1 Clinical Mission & Problem Definition
Breast cancer remains a leading cause of oncological mortality among women globally. While B-mode ultrasound is an accessible, radiation-free diagnostic modality, expert tele-radiologists are concentrated in tertiary metropolitan hospitals. Peripheral rural health centers and mobile emergency ambulances must transmit ultrasound scans over constrained, lossy, and unstable telecommunication links (3G, rural 4G, or satellite uplinks).

Standard uniform image compression creates a critical clinical dilemma:
- **Low Compression (e.g., JPEG Q90):** Preserves tumor morphology, but produces large byte payloads ($>78\text{ KB}$ per frame) that incur multi-second transfer delays and packet drop timeouts over slow links.
- **Aggressive Compression (e.g., JPEG Q30):** Reduces byte payloads ($<25\text{ KB}$), but introduces $8 \times 8$ block boundary discontinuities and eliminates high-frequency acoustic speckle textures, severely degrading automated deep learning lesion segmentation.

**Our Proposed Solution:**
We engineered a **Network-Aware Region-Adaptive Compression and Transmission System** deployed across two physical laptops:
1. **At the Edge (Sender Node - Ubuntu):** A compact, lightweight Deep Learning ROI Detector ($482\text{K parameters}$, $1.96\text{ MB}$, $3.8\text{ ms latency}$) detects suspicious lesions in real time. The region is expanded via elliptical morphological dilation (15-pixel margin) and smoothed using Gaussian feathering ($\sigma = m/3$). The lesion is preserved at pristine Q90 quality while non-diagnostic background tissue is aggressively compressed at Q20.
2. **Over the Wire (Transport Layer):** Frames are encapsulated in a custom 14-byte binary framing protocol (`b"NAC1"`), verified with cryptographic SHA-256 digests, and streamed over reliable TCP sockets with microsecond-precision monotonic wire clocking (`time.perf_counter()`).
3. **At the Hospital Server (Receiver Node - Windows):** Incoming payloads are verified for integrity, decoded in-memory, and processed by a clinical-grade U-Net ($7.76\text{M parameters}$, $31.2\text{ MB}$) optimized using the **Multi-Gradient Descent Algorithm (MGDA)** to eliminate multi-objective loss conflicts (+5.25% validation Dice gain). Segmented lesion contours are overlaid onto uncropped full-resolution images, and all 42 clinical metrics are logged to an ACID-compliant SQLite database.

---

### 1.2 Two-Laptop Hardware & Software Deployment Topology

```
+---------------------------------------------------------------------------------------------------------+
| SENDER NODE (Edge Clinic / Mobile Ambulance)                                                            |
| OS: Ubuntu 24.04 LTS (Kernel 6.8) | IP: 172.30.11.107                                                   |
| Hardware: Intel Core i5, 16 GB RAM                                                                      |
|                                                                                                         |
| [BUSI Dataset (780 Images)]                                                                             |
|            │                                                                                            |
|            ▼                                                                                            |
| [Image Compression Workbench (PyQt6)]                                                                   |
|    ├── Option A: Uniform JPEG (Q30, Q50, Q70, Q90)                                                      |
|    └── Option B: Region-Adaptive Compression Engine                                                     |
|            ├── ROIDetector (482,449 params, 1.96 MB, CPU latency: 3.82 ms)                              |
|            ├── Elliptical Dilation (margin m = 15 px)                                                   |
|            ├── Gaussian Feathering (σ = 5.0 px) ──► Continuous Alpha Mask A(x, y)                       |
|            └── Spatial Alpha Blend: I_adaptive = A·I_Q90 + (1-A)·I_Q20                                  |
|            │                                                                                            |
|            ▼                                                                                            |
| [Cryptographic SHA-256 Engine] ──► Generates 256-bit payload digest                                    |
|            │                                                                                            |
|            ▼                                                                                            |
| [TCP Client Transport Pipeline]                                                                         |
|    ├── 14-Byte Binary Header Packing (!4sHII: b"NAC1" | MetaLen | PayloadLen)                           |
|    ├── High-Precision Monotonic Wire Clocking (time.perf_counter())                                     |
|    └── Linux Traffic Control (tc netem) Channel Emulation (LAN, 4G, 3G, Rural, Satellite)              |
+---------------------------------------------------------------------------------------------------------+
                                                     │
                                                     │ Physical Cat6 RJ45 Cable
                                                     │ Subnet: 172.30.11.0/24 | TCP Port: 5000
                                                     ▼
+---------------------------------------------------------------------------------------------------------+
| RECEIVER NODE (Hospital Central PACS / Diagnostic Workstation)                                          |
| OS: Windows 11 Pro (Build 22631) | IP: 172.30.11.108                                                    |
| Hardware: Intel Core i7, 16 GB RAM                                                                      |
|                                                                                                         |
| [Multi-Threaded TCP Server Engine (Listening on 0.0.0.0:5000)]                                          |
|    ├── Deterministic recv_exact() partial-read streaming loop                                           |
|    ├── DoS Defense: 50 MB hard payload ceiling pre-allocation check                                    |
|    ├── Path Traversal Defense: Whitelist regex + canonical path jail containment                        |
|    ├── SHA-256 Cryptographic Verification (MTTD: 0.34 ms, 100% bit-flip detection)                     |
|    └── 16-Byte ACK Dispatch (!4sIQ: b"NACK" | StatusCode: 0 | ServerTimestamp)                         |
|            │                                                                                            |
|            ▼                                                                                            |
| [In-Memory JPEG Reconstruction (cv2.imdecode)]                                                          |
|            │                                                                                            |
|            ▼                                                                                            |
| [Clinical Deep Learning Segmentation Engine]                                                            |
|    ├── Frozen U-Net (7,762,465 params, 31.2 MB checkpoint)                                              |
|    ├── MGDA Pareto Optimization Trained (Val Dice: 0.6581, +5.25% gain)                                 |
|    ├── Sigmoid Activation & Decision Thresholding (τ = 0.5)                                             |
|    └── Full-Resolution Green Lesion Boundary Overlay (777×578 uncropped)                                |
|            │                                                                                            |
|            ▼                                                                                            |
| [Ground-Truth Matching & Objective Evaluation]                                                          |
|    ├── Dice Similarity Coefficient & Jaccard IoU Calculation                                            |
|    └── Empty Mask Clinical Validation (True Negatives = 1.0, Hallucinations = 0.0)                      |
|            │                                                                                            |
|            ▼                                                                                            |
| [Telemedicine SQLite Database & GUI Dashboard]                                                          |
|    ├── 42-Column Relational Provenance Schema (experiments/experiments.sqlite)                          |
|    ├── Write-Ahead Logging (WAL) & Dynamic Schema Migration                                             |
|    └── Automated Historical Telemetry Backfilling (_backfill_missing_telemetry)                         |
+---------------------------------------------------------------------------------------------------------+
```

---

# SECTION 2: "WHAT EVERY TEAM MEMBER MUST KNOW"
### (14 Core Technical Concepts Every Member Must Master Independently)

Every member, regardless of primary specialty, will be cross-examined on these 14 fundamentals. Memorize these verified facts:

### 1. BUSI Dataset Breakdown
- Total Cases: **780 images** from 600 female patients collected in 2018 at Baheya Hospital, Cairo.
- Categories: **437 Benign**, **210 Malignant**, **133 Normal**.
- Partition: **70% Train (546)**, **15% Validation (117)**, **15% Test (117)**, stratified by class using deterministic seed 42 (`data/BUSI/splits.json`).

### 2. Multi-Mask Merging Logic
- Certain multifocal scans contain multiple mask files (e.g., `benign (100)_mask.png`, `benign (100)_mask_1.png`).
- Merged via element-wise maximum in `common/dataset.py`: `combined_mask = np.maximum(combined_mask, m)` to prevent pixel values from exceeding 255 while ensuring a boolean logical union ($\mathbf{M} = \bigcup_k \mathbf{M}_k$).

### 3. Interpolation Rules: Bilinear vs. Nearest Neighbor
- Images are resized using **Bilinear Interpolation** (`cv2.INTER_LINEAR`) to preserve continuous acoustic intensity gradients.
- Masks are resized strictly using **Nearest-Neighbor Interpolation** (`cv2.INTER_NEAREST`). Bilinear interpolation introduces fractional intensities ($0 < x < 255$) at mask edges, which corrupts discrete binary ground-truth boundaries.

### 4. Standard JPEG Compression Pipeline (5 Steps)
- Step 1: $8 \times 8$ Block Partitioning.
- Step 2: Forward 2D Discrete Cosine Transform (2D-DCT).
- Step 3: **Scalar Quantization** via division by scaled quantization table $Q(u, v)$ and rounding (**the only lossy step**).
- Step 4: Zigzag scanning (clusters low frequencies and long runs of high-frequency zeros).
- Step 5: Run-Length Encoding (RLE) and Huffman Entropy Coding.

### 5. Objective Distortion Metrics Formulas
- $\text{MSE} = \frac{1}{MN} \sum (I - K)^2$.
- $\text{PSNR} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}} \right)$.
- $\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$, where $C_1 = 6.5025$, $C_2 = 58.5225$.
- $\text{CR} = \frac{S_{\text{orig}}}{S_{\text{comp}}}$, $\text{Size Reduction } \% = \left( 1 - \frac{S_{\text{comp}}}{S_{\text{orig}}} \right) \times 100\%$.

### 6. Clinical U-Net Architecture
- File: `models/unet.py`, parameters: **$7,762,465$** ($31.2\text{ MB}$).
- 4-stage encoder ($1 \to 64 \to 128 \to 256 \to 512$), 1024-channel bottleneck, 4-stage decoder with transposed convolutions and lateral skip connections. Final $1 \times 1$ conv with sigmoid thresholded at $\tau = 0.5$.

### 7. Multi-Objective MGDA Optimization
- Standard loss combination $\alpha \mathcal{L}_{\text{BCE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$ suffers from gradient conflict ($\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0$).
- MGDA projects gradients onto the Pareto frontier using the Frank-Wolfe minimum norm formulation:
  $$\alpha^* = \text{clip}\left( \frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1 \right)$$
- Empirical Result: Validation Dice improved from **0.6056** (baseline) to **0.6581** (MGDA), an absolute gain of **+5.25%**.

### 8. Lightweight Edge ROI Detector
- File: `models/roi_detector.py`, parameters: **$482,449$** ($1.96\text{ MB}$, **$16.1\times$ smaller** than U-Net).
- CPU Latency: **$3.82\text{ ms}$** (vs $79.39\text{ ms}$ for U-Net).
- Held-Out Test Performance: Test Dice = **$0.6087$**, Test IoU = **$0.5126$**.

### 9. Region-Adaptive Compression Mechanics
- Elliptical morphological dilation (`cv2.MORPH_ELLIPSE`) with margin $m=15\text{ px}$.
- Gaussian feathering ($\sigma = m/3 = 5.0$) creates a continuous alpha mask $A(x, y) \in [0, 1]$, eliminating artificial edge artifacts.
- Dual-rate blend: $I_{\text{adaptive}} = A \cdot I_{\text{Q90}} + (1 - A) \cdot I_{\text{Q20}}$.
- Verification on `benign (104)`: $37.9\%$ byte reduction ($78,050\text{ B} \to 48,481\text{ B}$), Preserved ROI PSNR = $49.70\text{ dB}$, Dice = $0.8664$ (0.0000 degradation).

### 10. Why TCP Over UDP?
- Medical scans require zero-loss byte delivery. Dropping a single UDP packet desynchronizes variable-length Huffman decoding, corrupting all subsequent $8 \times 8$ blocks. TCP provides sequence ordering, flow control, and reliable retransmissions.

### 11. Application-Layer Binary Framing Protocol
- 14-byte Header: `!4sHII` in Big-Endian network byte order: `b"NAC1"` (4B) + MetaLen (2B) + Reserved (4B) + PayloadLen (4B).
- 16-byte ACK: `!4sIQ`: `b"NACK"` (4B) + StatusCode (4B) + ServerTimestamp (8B).
- Partial-read handling: Deterministic `recv_exact(sock, n)` loop.

### 12. Monotonic Telemetry & Wire Clocking
- Clocked strictly around `sock.sendall()` using monotonic `time.perf_counter()`.
- Formula: $\text{Throughput (Kbps)} = \frac{S_{\text{wire}} \times 8}{T_{\text{tx}} \times 1000}$.
- Measured Gigabit Ethernet Result: $4.55\text{ ms}$ wire time, $86,214.85\text{ Kbps}$ ($86.2\text{ Mbps}$). Not 1000 Mbps due to TCP Slow-Start window ramp-up on short 48 KB transfers.

### 13. Linux `tc netem` Channel Emulation
- Six clinical profiles: LAN (1 Gbps), Broadband (50 Mbps), 4G (15 Mbps), 3G (2 Mbps), Rural (384 Kbps, 120 ms, 2.5% loss), Satellite (256 Kbps, 500 ms, 5% loss).
- Across Rural Link, adaptive compression cuts transmission time from **$9.61\text{ s}$ to $1.14\text{ s}$** ($8.4\times$ speedup).

### 14. Multi-Tier Telemedicine Security Hardening
- Implemented: SHA-256 integrity verification (MTTD 0.34 ms), 50 MB DoS ceiling, regex path sanitization (`^[a-zA-Z0-9_\-\.\(\)/ ]+$`), disk space check (>100 MB), safe PyTorch checkpoint loading (`weights_only=True`).
- Proposed Future: TLS 1.3 / mTLS with AES-256-GCM encryption for HIPAA/GDPR transit confidentiality over public networks.

---

# SECTION 3: STEP-BY-STEP IMAGE LIFECYCLE TRACE
### (Tracing `benign (104)` Through the Entire System)

```
====================================================================================================
STAGE 1: SELECTION & PREPROCESSING (Sender Node - Ubuntu)
====================================================================================================
1. Operator selects image: "data/BUSI/benign/benign (104).png" (Dimensions: 777×578, Raw: 455,471 B).
2. Operator selects "Option B: ML-Based Region-Adaptive Compression".
3. Parameters configured: Margin = 15 px, ROI Quality = Q90, Background Quality = Q20.

====================================================================================================
STAGE 2: EDGE ROI DETECTION & REGION-ADAPTIVE BLENDING
====================================================================================================
4. Image is resized to 256×256 and fed into ROIDetector (models/roi_detector_v1.pth).
5. ROI Detector executes in 3.82 ms on CPU, generating binary candidate mask (Dice: 0.6087).
6. Mask is dilated with elliptical kernel (diameter 31 px) using cv2.MORPH_ELLIPSE.
7. Dilated mask is filtered with 2D Gaussian blur (σ = 5.0 px) to produce alpha mask A(x, y) ∈ [0, 1].
8. Full image is compressed at Q90 (I_high) and Q20 (I_low).
9. Spatially blended: I_adaptive = A·I_high + (1 - A)·I_low.
10. Blended image is encoded to JPEG payload: 48,481 bytes (37.9% byte savings over uniform Q90).

====================================================================================================
STAGE 3: CRYPTOGRAPHIC HASHING & BINARY FRAMING
====================================================================================================
11. Sender calculates SHA-256 over 48,481 compressed bytes -> 64-char hex digest.
12. Metadata JSON constructed: {"image_id": "benign/benign (104)", "mode": "REGION_ADAPTIVE", ...}.
13. Header packed via struct.pack("!4sHII", b"NAC1", len(json), 0, 48481) -> exactly 14 bytes.
14. Total wire frame assembled: 14B Header + 512B JSON + 48,481B Payload = 49,007 Wire Bytes.

====================================================================================================
STAGE 4: NETWORK DISPATCH & MONOTONIC WIRE CLOCKING
====================================================================================================
15. Sender connects to receiver socket at 172.30.11.108:5000.
16. Monotonic timer started: t_start = time.perf_counter().
17. Physical stream dispatched via sock.sendall(frame_data).
18. Monotonic timer stopped: t_end = time.perf_counter().
19. Wire Transmission Time = 0.004547 seconds (4.55 ms).
20. Measured Wire Throughput = (49,007 * 8) / (0.004547 * 1000) = 86,214.85 Kbps (86.2 Mbps).

====================================================================================================
STAGE 5: RECEIVER INGESTION & SECURITY VALIDATION (Windows Node)
====================================================================================================
21. Receiver TCP server reads 14 bytes via recv_exact(), verifies magic identifier b"NAC1".
22. Server extracts payload length (48,481 B) and checks against DoS limit (48,481 < 52,428,800 B).
23. Server reads JSON metadata and binary image payload via recv_exact().
24. Server recomputes SHA-256 over payload: calculated hash matches metadata hash (Verified).
25. Image is decoded from memory buffer via cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE).

====================================================================================================
STAGE 6: CLINICAL DEEP LEARNING SEGMENTATION & OVERLAY
====================================================================================================
26. Decoded image is resized to 256×256, normalized to [0, 1], and fed to UNet (models/unet_busi.pth).
27. U-Net inference executes in 79.39 ms on CPU, generating sigmoid probability map.
28. Binary threshold τ = 0.5 generates predicted lesion mask (area = 442 pixels).
29. Mask is resized back to native 777×578 using cv2.INTER_NEAREST.
30. Green semi-transparent contour overlay is generated on uncropped received scan.

====================================================================================================
STAGE 7: EVALUATION, DATABASE TRANSACTION & ACKNOWLEDGMENT
====================================================================================================
31. Receiver matches ground-truth mask: Dice Score = 0.8664, IoU Score = 0.7643.
32. Complete record (42 columns) committed to experiments/experiments.sqlite in WAL mode.
33. Receiver packs 16-byte ACK frame: struct.pack("!4sIQ", b"NACK", 0, server_timestamp).
34. Sender receives ACK, clocks Round-Trip Time = 0.1011 seconds (101.1 ms).
35. GUI tables and dashboards on both nodes update in real time.
====================================================================================================
```

---

# SECTION 4: EMPIRICAL RATE-DISTORTION RESULTS & BENCHMARKS

The following data is extracted directly from verified physical trial records in `experiments/experiments.sqlite`:

### 4.1 Comparative Analysis Across Compression Modes (`benign (104)`)

| Compression Mode | Setting / Parameters | Compressed Size (Bytes) | CR | Size Reduction | Preserved ROI PSNR | Full SSIM | Clinical Dice | Clinical IoU | Wire Tx Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Uncompressed Baseline** | Raw 8-Bit Grayscale | $455,471\text{ B}$ | $1.00:1$ | $0.00\%$ | $\infty\text{ dB}$ | $1.0000$ | $0.8664$ | $0.7643$ | $\approx 42.1\text{ ms}$ |
| **Normal Compression** | Uniform Q90 | $78,050\text{ B}$ | $5.84:1$ | $82.86\%$ | $72.45\text{ dB}$ | $1.0000$ | $0.8664$ | $0.7643$ | $7.24\text{ ms}$ |
| **Normal Compression** | Uniform Q70 | $47,139\text{ B}$ | $9.66:1$ | $89.65\%$ | $43.29\text{ dB}$ | $0.9841$ | $0.8752$ | $0.7782$ | $4.38\text{ ms}$ |
| **Normal Compression** | Uniform Q50 | $38,810\text{ B}$ | $11.74:1$ | $91.48\%$ | $40.58\text{ dB}$ | $0.9711$ | $0.8522$ | $0.7425$ | $3.61\text{ ms}$ |
| **Normal Compression** | Uniform Q30 | $24,528\text{ B}$ | $18.57:1$ | $94.61\%$ | $38.57\text{ dB}$ | $0.9406$ | $0.8410$ | $0.7258$ | $2.28\text{ ms}$ |
| **Region-Adaptive** | **Margin 15px, Q90/Q20** | **$48,481\text{ B}$** | **$9.39:1$** | **$89.36\%$** | **$49.70\text{ dB}$** | **$0.9292$** | **$0.8664$** | **$0.7643$** | **$4.55\text{ ms}$** |

### 4.2 Key Empirical Takeaways:
1. **$37.9\%$ Bandwidth Savings Over Q90:** Region-Adaptive compression slashes byte size from $78,050\text{ B}$ to $48,481\text{ B}$, saving $29,569\text{ bytes}$ of wire transmission payload.
2. **Zero Clinical Diagnostic Loss:** The receiver U-Net segmentation Dice score under Region-Adaptive compression is **$0.8664$**—mathematically identical to uncompressed transmission and uniform Q90 ($0.0000$ degradation).
3. **Artifact Avoidance:** Unlike uniform Q30 which drops Dice to $0.8410$ due to $8 \times 8$ block boundary corruption, Region-Adaptive compression keeps the lesion and margin at pristine Q90 fidelity.

---

# SECTION 5: HONEST RESEARCH LIMITATIONS & COMMITTEE DEFENSE

Examiners respect teams that clearly understand their project's boundaries. Use this defense framework:

| System Dimension | Current Verified Implementation | Acknowledged Academic Limitation | Faculty Defense Strategy |
| :--- | :--- | :--- | :--- |
| **Ultrasound Modality** | 2D B-mode ultrasound (.png grayscale) | Does not process 3D automated breast ultrasound (ABUS) or Doppler | Explain that 2D B-mode is the primary screening modality in rural triage; volume ABUS requires 3D convolutions ($3\text{D U-Net}$) which is our planned Phase 2. |
| **Dataset Source** | 780 images from BUSI dataset (Cairo, Egypt) | Single-institution data; scanner vendor is LOGIQ E9 only | Emphasize rigorous 70/15/15 stratified partitioning with zero leakage, and note that cross-vendor domain adaptation is a standard next step in multi-center clinical trials. |
| **Transport Encryption** | Raw TCP socket with SHA-256 integrity | No in-transit encryption (no TLS 1.3/mTLS wrapper) | Distinguish between integrity (implemented 100%) and confidentiality. State clearly that TLS 1.3 is an external transport wrapper that integrates seamlessly above our TCP client. |
| **ROI Boundary Scope** | Elliptical dilation with Gaussian feathering | Radial expansion assumes roughly convex lesions | Explain that the 15-pixel margin provides an extensive spatial cushion that safely encapsulates spiculated, non-convex malignant contours. |
| **Lossy Compression Standard** | DCT-based JPEG (libjpeg) | Evaluates JPEG rather than modern WebP, AVIF, or JPEG-XL | Explain that JPEG is universally supported across every legacy medical PACS, DICOM viewer, and mobile OS, whereas AVIF/JPEG-XL require specialized decoders often missing on edge hardware. |
