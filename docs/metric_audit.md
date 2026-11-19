# Telemedicine Research Metric Audit & Measurement Specification

**Project**: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation  
**Document Version**: 1.1.0  
**Compliance**: Strict Zero-Dummy-Data Policy — Explicit demarcation between Measured Physical Metrics, Configured Test Parameters, and Theoretical Reference Bounds.

---

## 1. Executive Summary

This metric audit specifies the mathematical definitions, data sources, measurement mechanisms, precision, and physical units for all 42 telemetry parameters recorded in `experiments/experiments.sqlite` and `experiments/experiments.csv`.

Under our experimental framework:
- **No theoretical estimates are substituted for measured values.**
- **Pure wire transmission duration is strictly isolated from application round-trip latency (RTT).**
- **Sender-side and receiver-side processing timings are captured independently via high-resolution monotonic clocks (`time.perf_counter()`).**
- **Segmentation metrics (Dice & IoU) are computed only against authenticated ground-truth annotations.**

---

## 2. Complete Metric Specification Table

| # | Column / Metric Name | Category | Classification | Formula / Definition | Measurement Method / Source | Unit |
|---|----------------------|----------|----------------|----------------------|-----------------------------|------|
| 1 | `experiment_id` | Identifier | Generated | Unique string identifier | `f"{PREFIX}_{timestamp_ms}_Q{quality}"` | String |
| 2 | `timestamp` | Telemetry | Measured | Wall-clock start of experiment | `datetime.now().strftime("%Y-%m-%d %H:%M:%S")` | ISO 8601 |
| 3 | `image_id` | Metadata | Ground Truth | Unique image filename/ID | BUSI filename (e.g., `benign (10)`) | String |
| 4 | `image_category` | Metadata | Ground Truth | Clinical category | Extracted from BUSI directory (`benign`, `malignant`, `normal`) | Categorical |
| 5 | `compression_method` | Config | Configured | Compression algorithm mode | User GUI selection (`NORMAL` vs `REGION_ADAPTIVE`) | Categorical |
| 6 | `compression_quality` | Config | Configured | Base JPEG compression factor | Quantization scale $Q \in [1, 100]$ | Integer |
| 7 | `compression_params` | Config | Configured | Auxiliary parameters | JSON descriptor: `{"roi_q": 90, "bg_q": 30, "margin_px": 15}` | JSON String |
| 8 | `region_adaptive_margin` | Config | Configured | Lesion dilation expansion margin | Pixel radius $m$ applied to ROI mask via `cv2.dilate()` | Pixels |
| 9 | `compression_model_version` | Metadata | Architecture | Compression / ROI model tag | Version string (e.g. `roi_detector_v1` or `manual_pilot`) | String |
| 10 | `original_file_size` | Image | Measured | Byte length of raw uncompressed/PNG file | `Path.stat().st_size` or `len(uncompressed_bytes)` | Bytes |
| 11 | `compressed_file_size` | Image | Measured | Byte length of compressed JPEG / fused payload | `len(compressed_bytes)` | Bytes |
| 12 | `transmitted_bytes` | Network | Measured | Total frame bytes sent over TCP socket | $\text{Header} + \text{Metadata JSON} + \text{Image Bytes}$ | Bytes |
| 13 | `received_bytes` | Network | Measured | Total frame bytes read from socket by receiver | `recv_image_frame()` stream accumulator count | Bytes |
| 14 | `compression_ratio` | Compression | Calculated | $CR = \frac{\text{original\_file\_size}}{\text{compressed\_file\_size}}$ | Direct arithmetic division | Ratio ($:1$) |
| 15 | `size_reduction_pct` | Compression | Calculated | $\Delta \% = \frac{\text{orig} - \text{comp}}{\text{orig}} \times 100$ | Relative size savings percentage | Percentage (%) |
| 16 | `psnr` | Fidelity | Measured | $10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$ | Peak Signal-to-Noise Ratio via `cv2.PSNR()` | Decibels (dB) |
| 17 | `ssim` | Fidelity | Measured | Structural Similarity Index (Wang et al.) | Multi-scale structural index via `skimage.metrics.structural_similarity()` | Scalar $[0.0, 1.0]$ |
| 18 | `bandwidth_limit` | Network | Configured | Emulated network bandwidth ceiling | `tc qdisc add dev <iface> root netem rate <R>` | String (e.g. `2mbit`) |
| 19 | `packet_loss_rate` | Network | Configured | Emulated packet dropping rate | `tc qdisc ... loss <P>%` | Percentage (%) |
| 20 | `network_delay_ms` | Network | Configured | Emulated one-way link latency | `tc qdisc ... delay <D>ms <J>ms` | Milliseconds (ms) |
| 21 | `theoretical_transmission_time_sec` | Benchmark | Theoretical | $T_{\text{theo}} = \frac{\text{compressed\_bytes} \times 8}{R_{\text{bps}}}$ | Theoretical wire delay at configured bandwidth $R_{\text{bps}}$ | Seconds (s) |
| 22 | `transmission_time_sec` | Network | Measured | Pure socket byte transmission duration | $t_{\text{wire\_end}} - t_{\text{wire\_start}}$ around `sendall()` | Seconds (s) |
| 23 | `measured_throughput_kbps` | Network | Measured | $R_{\text{meas}} = \frac{\text{transmitted\_bytes} \times 8}{t_{\text{wire}} \times 1000}$ | Physical wire transmission rate | Kilobits/sec (Kbps) |
| 24 | `connection_time_ms` | Network | Measured | Socket handshake duration | $t_{\text{connected}} - t_{\text{connect\_start}}$ | Milliseconds (ms) |
| 25 | `round_trip_time_sec` | Network | Measured | End-to-end transport RTT | $t_{\text{ack\_received}} - t_{\text{send\_start}}$ | Seconds (s) |
| 26 | `effective_bandwidth_kbps` | Network | Measured | $R_{\text{eff}} = \frac{\text{transmitted\_bytes} \times 8}{\text{RTT} \times 1000}$ | Application-level effective bandwidth | Kilobits/sec (Kbps) |
| 27 | `decoding_time_ms` | Processing | Measured | Image decompression duration | Receiver-side $t_{\text{dec\_end}} - t_{\text{dec\_start}}$ of `cv2.imdecode()` | Milliseconds (ms) |
| 28 | `verification_time_ms` | Processing | Measured | Cryptographic verification duration | Receiver-side `hashlib.sha256(bytes).hexdigest()` timing | Milliseconds (ms) |
| 29 | `inference_time_ms` | Processing | Measured | U-Net deep learning execution duration | Receiver-side `torch.cuda` or CPU inference pass duration | Milliseconds (ms) |
| 30 | `end_to_end_time_ms` | Processing | Measured | Total client-to-inference pipeline time | $t_{\text{post\_eval}} - t_{\text{pre\_compression}}$ | Milliseconds (ms) |
| 31 | `checksum_verified` | Security | Measured | SHA-256 payload integrity confirmation | `receiver_sha256 == metadata_sha256` | Boolean ($1$ or $0$) |
| 32 | `dice_coefficient` | Clinical | Measured | $\frac{2 \lvert X \cap Y \rvert + \epsilon}{\lvert X \rvert + \lvert Y \rvert + \epsilon}$ | Evaluated against ground-truth mask ($256 \times 256$) | Scalar $[0.0, 1.0]$ |
| 33 | `iou` | Clinical | Measured | $\frac{\lvert X \cap Y \rvert + \epsilon}{\lvert X \cup Y \rvert + \epsilon}$ | Jaccard index against ground-truth mask | Scalar $[0.0, 1.0]$ |
| 34 | `ground_truth_id` | Clinical | Ground Truth | Filename of matched ground-truth mask | BUSI mask identifier (e.g. `benign (10)_mask.png`) | String |
| 35 | `experiment_status` | Status | Operational | Trial execution outcome | `SUCCESS`, `CHECKSUM_MISMATCH`, `TIMEOUT`, `SOCKET_ERROR` | Categorical |
| 36 | `error_message` | Status | Operational | Error diagnosis string | Exception traceback or rejection message | String or `None` |
| 37 | `software_version` | Metadata | Operational | System software version tag | Git tag or application version (`1.1.0`) | String |
| 38 | `model_version` | Metadata | Operational | Segmentation model weights version | Checkpoint tag (`unet_mgda_v1` or `unet_baseline_v1`) | String |
| 39 | `transmission_mode` | Config | Configured | Execution dispatch mode | `MANUAL`, `BATCH_CONTROLLED`, `ADAPTIVE_AUTOMATED` | Categorical |
| 40 | `artifact_path` | Storage | Operational | Filepath of saved segmentation overlay | Relative path to PNG overlay in `data/received/` | Filepath |
| 41 | `notes` | Telemetry | Narrative | Operational observations | Human-readable notes and emulation parameters | String |
| 42 | `id` | Storage | Primary Key | Database autoincrement ID | SQLite sequence integer | Integer |

---

## 3. Discrepancy Prevention & Audit Verification

Prior to this upgrade, `transmission_time_sec` conflated:
1. Pure wire transmission time over the socket.
2. Receiver disk write time.
3. Receiver SHA-256 computation time.
4. Return ACK latency.

### Audit Remediation:
1. **Wire Time (`transmission_time_sec` / `wire_time_sec`)**: Measures exclusively the time spent by `send_image_frame()` pushing payload bytes through the local TCP buffer.
2. **RTT (`round_trip_time_sec`)**: Separately measures total round-trip time from `send_image_frame()` start until `recv_ack_frame()` completes.
3. **Receiver Timers (`decoding_time_ms`, `verification_time_ms`)**: Measured directly on the receiver node and returned inside the TCP acknowledgment frame (`AcknowledgmentMessage`).
4. **Theoretical Bandwidth Separation**: Theoretical transmission time ($T_{\text{theo}}$) is logged to its own dedicated column (`theoretical_transmission_time_sec`) and is never assigned to `transmission_time_sec`.
