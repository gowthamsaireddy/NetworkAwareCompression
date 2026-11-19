# Comprehensive Repository Audit & Architectural Implementation Plan
## Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

**Document Version**: 1.0.0  
**Date**: November 2026  
**Target Environment**: 
- **Sender Node**: Ubuntu Linux (tc/netem traffic control, network interface detection, dual-mode compression, TCP client, PyQt6 GUI)
- **Receiver Node**: Windows 11 Host (TCP server daemon, SHA-256 integrity check, frozen U-Net inference, Dice/IoU evaluation, PyQt6 GUI)

---

# SECTION 1: COMPREHENSIVE REPOSITORY AUDIT

### 1.1 Existing Functionality & Working Components

| Component / Module | Implementation Location | Current Operational Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Dataset Indexing & Manifest** | `sender/dataset_manager.py` | **Fully Functional** | Recursively indexes 780 BUSI scans (`benign`, `malignant`, `normal`), pairs multi-lesion masks, generates `splits.json` (546 train, 117 val, 117 test). |
| **Normal JPEG Compression** | `sender/compression.py` | **Fully Functional** | In-memory variable JPEG compression (Q30, Q50, Q70, Q90), enforces strict mask protection (raises error if mask passed), computes PSNR and SSIM. |
| **Image Quality Metrics** | `sender/image_metrics.py` | **Fully Functional** | Genuine PSNR (dB) and SSIM calculation using `skimage.metrics`. |
| **TCP Framing Protocol** | `shared/protocol.py`, `shared/metadata.py` | **Fully Functional** | Binary protocol `NAC1`: 14-byte image header, UTF-8 metadata JSON, raw payload bytes. 10-byte ACK header `NACK`. Chunked streaming (64 KB). |
| **TCP Sender Client** | `sender/tcp_client.py` | **Fully Functional** | Connects to receiver, sends framed payload, awaits verified ACK, logs monotonic transfer duration and measured throughput. |
| **TCP Receiver Server** | `receiver/tcp_server.py` | **Fully Functional** | Multi-threaded listening daemon on port 5000, receives binary frames, performs SHA-256 verification, writes verified images to `data/received/`. |
| **Integrity Verification** | `receiver/integrity.py` | **Fully Functional** | Cryptographic SHA-256 checksum matching and payload byte length verification before issuing ACK. |
| **Frozen U-Net Segmentation** | `receiver/segmentation.py`, `models/unet.py` | **Fully Functional** | 7.76M parameter U-Net mounted in frozen mode (`model.eval()`, `torch.no_grad()`), resizes to $256 \times 256$, thresholded at 0.5. |
| **Segmentation Overlap Metrics** | `receiver/segmentation_metrics.py` | **Fully Functional** | Exact mathematical Dice Similarity Coefficient (DSC) and IoU (Jaccard Index) with Empty Mask Policy. |
| **Trained Model Checkpoints** | `models/best_unet.pth`, `models/best_unet_mgda.pth` | **Fully Functional** | Baseline model (Epoch 21, Val Dice 0.6056) and MGDA model (Epoch 19, Val Dice 0.6581) preserved. |
| **Experiment Results Storage** | `common/results_manager.py` | **Working (Baseline)** | SQLite (`experiments/experiments.sqlite`, 27 records) and CSV (`experiments/experiments.csv`). |
| **Network Emulation** | `sender/network_control.py` | **Partially Functional** | Linux `tc/netem` command generation for rate, delay, loss. |
| **PyQt6 Desktop GUIs** | `ui/sender/`, `ui/receiver/` | **Fully Functional** | Complete 7-tab GUI on both Sender and Receiver. |

---

### 1.2 Metric Calculation Audit: Theoretical vs. Measured

A rigorous audit of the existing metric calculation pipelines reveals important distinctions between theoretical estimations and direct physical measurements:

| Metric Name | Stored Field in DB | Exact Source Formula | Measurement Methodology | Unit | Classification | Audit Finding / Discrepancy Identified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original File Size** | `original_file_size` | `Path.stat().st_size` or `len(raw_bytes)` | Python OS file stat on source image | Bytes | **Measured** | Accurate. |
| **Compressed File Size** | `compressed_file_size` | `len(out_stream.getvalue())` | Measured length of JPEG binary buffer | Bytes | **Measured** | Accurate. |
| **Compression Ratio** | `compression_ratio` | $\text{orig\_size} / \text{comp\_size}$ | Direct arithmetic quotient | Ratio (:1) | **Calculated** | Accurate. |
| **Size Reduction %** | (Present in UI) | $(1 - \text{comp\_size} / \text{orig\_size}) \times 100$ | Direct arithmetic percentage | % | **Calculated** | Missing from SQLite schema; needs persistent column. |
| **PSNR** | `psnr` | $10 \log_{10} \frac{255^2}{\text{MSE}}$ | `skimage.metrics.peak_signal_noise_ratio` | dB | **Measured** | Accurate. Returns $\infty$ for identical images. |
| **SSIM** | `ssim` | Structural similarity algorithm | `skimage.metrics.structural_similarity` | $[-1, 1]$ | **Measured** | Accurate. Multi-channel axis properly specified. |
| **Transmission Duration** | `transmission_time_sec` | $t_{\text{end}} - t_{\text{start}}$ | Hardware monotonic clock `time.perf_counter()` | Seconds | **Measured** | **AUDIT DISCREPANCY**: Currently spans `send_image_frame()` **plus** `recv_ack_frame()`. This measures **Application-Level Round-Trip Time (RTT + Processing)**, including receiver disk write, SHA-256 hashing, and ACK round trip, rather than pure network wire transmission time. |
| **Measured Throughput** | `measured_throughput_kbps` | $\frac{\text{Total Bytes} \times 8}{t_{\text{elapsed}} \times 1000}$ | Arithmetic quotient of sent bytes over measured time | Kbps | **Measured** | Derived from the application RTT duration above. While empirical and genuine, it reflects application effective throughput rather than raw physical link bandwidth. |
| **Theoretical Wire Time** | `theoretical_transmission_time_sec` | $\frac{\text{Bytes} \times 8}{\text{Bandwidth (bps)}}$ | Idealized wire bit clock | Seconds | **Theoretical** | Purely theoretical based on configured network profile rate (e.g. 512 Kbps). Handled as nullable. |
| **Connection Time** | (Not stored) | $t_{\text{connect\_end}} - t_{\text{connect\_start}}$ | Socket `connect()` duration | ms | **Missing** | Not currently measured or persisted in DB. |
| **Inference Duration** | (In UI / log) | $t_{\text{infer\_end}} - t_{\text{infer\_start}}$ | Hardware monotonic clock around U-Net forward pass | ms | **Measured** | Computed in `receiver/segmentation.py`, but **omitted from `experiments.sqlite` record**. |
| **Integrity Verification Time**| (Not stored) | $t_{\text{sha\_end}} - t_{\text{sha\_start}}$ | Monotonic timer around `hashlib.sha256()` | ms | **Missing** | Not currently timed or logged. |
| **Image Decoding Time** | (Not stored) | $t_{\text{decode\_end}} - t_{\text{decode\_start}}$ | Monotonic timer around `cv2.imdecode()` | ms | **Missing** | Not currently timed or logged. |
| **Dice Similarity (DSC)** | `dice_coefficient` | $\frac{2 \|P \cap G\|}{\|P\| + \|G\|}$ | Pixel-level overlap on $256 \times 256$ binary masks | $[0.0, 1.0]$ | **Calculated** | Accurate. Enforces Empty Mask Policy. |
| **Intersection over Union** | `iou` | $\frac{\|P \cap G\|}{\|P \cup G\|} = \frac{\text{Dice}}{2 - \text{Dice}}$ | Pixel-level union on $256 \times 256$ binary masks | $[0.0, 1.0]$ | **Calculated** | Accurate. Enforces Empty Mask Policy. |

---

### 1.3 Communication Protocol Audit (`shared/protocol.py` & `shared/metadata.py`)

1. **Wire Framing Stability**:
   - The 14-byte `IMAGE_HEADER_FORMAT = "!4sHII"` (`b"NAC1"`, version, metadata length $N$, image payload length $M$) is clean and robust.
   - It supports variable-length metadata JSON up to 64 KB and payload up to 50 MB with strict bounds checking.
2. **Backward Compatibility Strategy**:
   - `TransmissionMetadata` currently requires exact fields in `from_dict`.
   - To support **ML-Based Region-Adaptive Compression** without breaking the wire protocol:
     - New fields (`compression_mode`: `"NORMAL"` or `"REGION_ADAPTIVE"`, `adaptive_margin`: `int`, `model_version`: `str`, `roi_quality`: `int`, `bg_quality`: `int`) will be added with backward-compatible defaults (`dict.get(key, default)`).
     - The binary frame structure (`NAC1` + uint16 + uint32 + uint32 + JSON + Bytes) will remain **strictly identical**, ensuring 100% wire-compatibility with existing receivers.

---

### 1.4 Result Storage System Audit (`common/results_manager.py`)

1. **Current State**:
   - Database: `experiments/experiments.sqlite` (27 valid experiments recorded).
   - CSV: `experiments/experiments.csv`.
   - Existing table schema has 20 columns.
2. **Deficiencies vs. Research Requirements**:
   - Lacks transactional safety guarantees (`BEGIN IMMEDIATE` / auto-rollback on failure).
   - Missing fields: `image_category`, `compression_method`, `compression_params`, `region_adaptive_margin`, `compression_model_version`, `transmitted_bytes`, `received_bytes`, `size_reduction_pct`, `connection_time_ms`, `decoding_time_ms`, `verification_time_ms`, `inference_time_ms`, `end_to_end_time_ms`, `experiment_status`, `error_message`, `software_version`, `model_version`, `artifact_path`.
   - Lacks automated database backup mechanisms.
   - Lacks disk space monitoring and temporary file cleanup controls.

---

### 1.5 Missing & Partially Implemented Features Summary

1. **Separate Compression Options in GUI**:
   - Currently, the GUI has a single JPEG compression view and an "Adaptive View" that only modulates global JPEG quality based on bandwidth.
   - Missing: A clear, explicit dual-mode selection: **Option A: Normal Compression** vs. **Option B: ML-Based Region-Adaptive Compression**, with distinct control panels and parameter configurations.
2. **ML-Based Region-Adaptive Compression Engine**:
   - Missing: Sender-side lesion localization/detection model, morphological dilation margin expansion, feathered blending, and dual-quality compression.
3. **Controlled & Manual Experimental Execution**:
   - Missing: Image selector (single/multiple), estimated experiment count and duration display, mandatory confirmation modal, pause/resume/cancel controls, and deduplication/resume logic.
4. **Network Emulation Enhancements**:
   - Missing: Mandatory user confirmation modal before applying `tc` rules, post-experiment guaranteed cleanup (`sudo tc qdisc del`), and separation of configured profile parameters from measured telemetry.
5. **Security Framework & Testing**:
   - Missing: Automated security test harness (corrupt payloads, checksum mismatch, unauthorized connection, malformed messages), security metrics (MTTD, MTTR, MTBF, integrity verification overhead), and TLS/Authentication implementation architecture.
6. **Research Visualization & Limitations Documentation**:
   - Missing: Comparative plotting tools for Region-Adaptive vs. Normal JPEG across cohorts, and formal `docs/limitations.md`.

---

# SECTION 2: ARCHITECTURAL DESIGN & IMPLEMENTATION BLUEPRINT

## 2.1 Architecture for ML-Based Region-Adaptive Compression (Option B)

```
+----------------------------------------------------------------------------------------------------+
|                                    SENDER NODE (UBUNTU LINUX)                                      |
|                                                                                                    |
|  Raw Ultrasound Frame                                                                              |
|  (256 x 256 Grayscale) ───────+                                                                    |
|                               │                                                                    |
|                               ▼                                                                    |
|                 [ Lightweight Lesion ROI Detector ]                                                |
|                 (Trained on BUSI Train Split Only)                                                 |
|                               │                                                                    |
|                               ▼                                                                    |
|                  Raw Lesion Mask / Bounding Area                                                   |
|                               │                                                                    |
|                               ▼                                                                    |
|                 [ Morphological Margin Dilation ]                                                  |
|                 (Radius: m = 5, 10, 15, 20 pixels)                                                 |
|                               │                                                                    |
|                               ▼                                                                    |
|                 [ Alpha Feathering (Gaussian Blur) ]                                               |
|                 (Prevents sharp artificial boundaries)                                             |
|                               │                                                                    |
|                               ▼                                                                    |
|          +────────────────────┴────────────────────+                                               |
|          │                                         │                                               |
|          ▼                                         ▼                                               |
|   [ Region of Interest ]                   [ Background Tissue ]                                   |
|   High Quality (Q90 or Lossless)           Aggressive Compression (Q20 - Q30)                      |
|          │                                         │                                               |
|          +────────────────────┬────────────────────+                                               |
|                               │                                                                    |
|                               ▼                                                                    |
|                 [ Feathered Spatial Compositing ]                                                  |
|                 Fused Compressed Image (Lower Total Size)                                          |
|                               │                                                                    |
|                               ▼                                                                    |
|                 [ TCP Streaming Frame (b"NAC1") ] ──────────────────────────► (Windows Receiver)   |
+----------------------------------------------------------------------------------------------------+
```

### Technical Design Choices:
1. **Model Selection**:
   - A lightweight `MobileNetV3-UNet` or `Tiny-UNet` (~350K parameters) trained strictly on `splits.json["train"]`.
   - In manual/pilot mode, the system also supports ground-truth-guided simulation to establish the theoretical upper bound of region-adaptive compression.
2. **Artifact Elimination**:
   - Direct rectangular cropping causes high-frequency ringing artifacts at the boundary that corrupt U-Net convolutional feature maps.
   - We apply a normalized Gaussian alpha mask: $I_{\text{comp}} = \alpha \cdot I_{\text{high}} + (1 - \alpha) \cdot I_{\text{low}}$, ensuring smooth gradient transitions between the preserved lesion region and the compressed acoustic background.

---

## 2.2 Controlled Experimental Execution Engine

To prevent uncontrolled bulk runs across all 780 images, the execution engine will enforce:
1. **Interactive Image Selection**: Select 1 image, multiple specific images, or category slices (`benign`, `malignant`, `normal`).
2. **Pre-Execution Confirmation Dialog**:
   - Displays: Selected images count ($N$), selected qualities ($Q$), selected network profiles ($P$), **Total Planned Trials ($N \times Q \times P$)**, and **Estimated Duration**.
   - Requires explicit user click on `"Confirm & Execute"`.
3. **Execution Control**:
   - State machine: `IDLE` $\to$ `RUNNING` $\to$ `PAUSED` $\to$ `CANCELLED` $\to$ `COMPLETED`.
   - Sequential execution: exactly one socket transfer and one inference at a time.
   - Resumption support: skips already completed `(image_id, quality, profile, method)` combinations.

---

## 2.3 Comprehensive Storage Schema Upgrade (Zero Data Loss)

We will upgrade `experiments/experiments.sqlite` via non-destructive `ALTER TABLE` commands, preserving all 27 existing records while adding the 18 new research fields:

```sql
-- Schema Migration (Preserves all existing rows):
ALTER TABLE experiments ADD COLUMN image_category TEXT;
ALTER TABLE experiments ADD COLUMN compression_method TEXT DEFAULT 'Normal JPEG';
ALTER TABLE experiments ADD COLUMN compression_params TEXT;
ALTER TABLE experiments ADD COLUMN region_adaptive_margin INTEGER;
ALTER TABLE experiments ADD COLUMN compression_model_version TEXT;
ALTER TABLE experiments ADD COLUMN size_reduction_pct REAL;
ALTER TABLE experiments ADD COLUMN transmitted_bytes INTEGER;
ALTER TABLE experiments ADD COLUMN received_bytes INTEGER;
ALTER TABLE experiments ADD COLUMN connection_time_ms REAL;
ALTER TABLE experiments ADD COLUMN decoding_time_ms REAL;
ALTER TABLE experiments ADD COLUMN verification_time_ms REAL;
ALTER TABLE experiments ADD COLUMN inference_time_ms REAL;
ALTER TABLE experiments ADD COLUMN end_to_end_time_ms REAL;
ALTER TABLE experiments ADD COLUMN experiment_status TEXT DEFAULT 'SUCCESS';
ALTER TABLE experiments ADD COLUMN error_message TEXT;
ALTER TABLE experiments ADD COLUMN software_version TEXT DEFAULT '2.0.0';
ALTER TABLE experiments ADD COLUMN model_version TEXT DEFAULT 'best_unet_mgda';
ALTER TABLE experiments ADD COLUMN artifact_path TEXT;
```

---

## 2.4 Security Evaluation & Testing Framework

### A. Time Metrics
1. **Mean Time to Detect (MTTD)**: $\text{MTTD} = \frac{1}{N_{\text{attacks}}} \sum (t_{\text{detected}} - t_{\text{injected}})$.
2. **Mean Time to Recover (MTTR)**: $\text{MTTR} = \frac{1}{N_{\text{failures}}} \sum (t_{\text{recovered}} - t_{\text{failed}})$.
3. **Mean Time Between Failures (MTBF)**: $\text{MTBF} = \frac{\text{Total Operating Time}}{N_{\text{failures}}}$.

### B. Efficiency Metrics
1. **Integrity Verification Overhead**: $\text{Overhead} = \frac{t_{\text{SHA256}}}{t_{\text{total}}} \times 100\%$.
2. **Successful Transmission Rate**: $\text{STR} = \frac{N_{\text{ACK\_OK}}}{N_{\text{total\_transfers}}} \times 100\%$.
3. **Automated Recovery Success Rate**: $\text{RSR} = \frac{N_{\text{successful\_reconnects}}}{N_{\text{disconnections}}} \times 100\%$.

### C. Security Testing Suite (`tests/test_security_resilience.py`)
- Test 1: Payload Bit Flips $\to$ Verifies `ACK_CHECKSUM_MISMATCH` is emitted.
- Test 2: Truncated Payload $\to$ Verifies `recv_exact` socket timeout and graceful abort.
- Test 3: Path Traversal Attack (`../../evil.jpg`) $\to$ Verifies path sanitization raises `ValueError`.
- Test 4: Oversized Frame Header ($> 50 \text{ MB}$) $\to$ Verifies DoS boundary rejection.
- Test 5: Abrupt Connection Reset $\to$ Verifies sender reconnection backoff without GUI crash.

---

# SECTION 3: STEP-BY-STEP PHASED IMPLEMENTATION PLAN

| Phase | Core Objective | Key Deliverables & Files | User Confirmation Point |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Repository Audit & Plan** | Comprehensive audit report, metric formulas, and architectural plan (`docs/enhancement_audit_and_implementation_plan.md`). | **CURRENT STAGE: Review & Approve Plan** |
| **Phase 2** | **Research Infrastructure & Metric Audit** | Schema migration for `experiments.sqlite`, transactional writes, backup utility, exact physical timers for connection, decoding, inference, verification, and transfer. | Proceed to Phase 3 |
| **Phase 3** | **Dual-Mode Compression GUI** | Update `ui/sender/compression_view.py` and `transmission_view.py` with separate Option A (Normal JPEG) and Option B (ML Region-Adaptive) buttons and parameters. | Proceed to Phase 4 |
| **Phase 4** | **ML-Based Region-Adaptive Compression** | Implement `sender/region_adaptive_compressor.py`, ROI locator, dilation margin expansion, feathered blending, and validate against train/val split. | Proceed to Phase 5 |
| **Phase 5** | **Controlled Experimental Execution Engine** | Implement manual image selector, confirmation dialog with planned trials and runtime estimates, pause/resume/cancel controls, and deduplication logic. | Proceed to Phase 6 |
| **Phase 6** | **Network Emulation & Guaranteed Cleanup** | Integrate Ubuntu `tc/netem` profiles with user confirmation modal, post-trial guaranteed reset (`del dev root`), and parameter logging. | Proceed to Phase 7 |
| **Phase 7** | **Security Evaluation & Testing Suite** | Implement security evaluation harness (`tests/test_security_resilience.py`), evaluate MTTD, MTTR, MTBF, integrity overhead, and document TLS architecture. | Proceed to Phase 8 |
| **Phase 8** | **Research Visualization & Limitations Doc** | Generate publication-ready comparison plots (`Normal vs Region-Adaptive`, sub-cohorts), export summary tables, and author `docs/limitations.md`. | Final Review & Wrap-up |

---

# SECTION 4: VERIFICATION INSTRUCTIONS (TWO LAPTOPS)

### Ubuntu Sender Node Verification Checklist:
1. `python main_sender.py` launches the GUI cleanly.
2. Select an image from `Dataset Management`.
3. In `Compression & Quality`:
   - Click **Option A (Normal Compression)** $\to$ Verifies JPEG Q30..Q90.
   - Click **Option B (ML Region-Adaptive)** $\to$ Verifies lesion ROI preservation + margin expansion.
4. Set network profile in `Network Emulation` $\to$ Confirm rule application and verify cleanup.
5. Transmit image via TCP $\to$ Verifies chunked transfer, SHA-256 calculation, and ACK reception.

### Windows Receiver Node Verification Checklist:
1. `python main_receiver.py` launches the GUI cleanly.
2. Click `Start Server` on port 5000 $\to$ Listening daemon starts.
3. Upon image reception:
   - Verifies SHA-256 digest match.
   - Emits `ACK_OK`.
   - Runs frozen U-Net inference on GPU/CPU.
   - Evaluates Dice and IoU against ground-truth mask.
   - Commits transaction to SQLite and CSV.
