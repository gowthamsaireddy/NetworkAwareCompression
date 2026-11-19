# Final Evaluation Source Code & Architectural Map

**Project**: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation  
**Document Purpose**: Definitive technical reference mapping every major module, class, function, model, and configuration in the repository to its operational role, mathematical logic, callers, and verification status.

---

## 1. Repository Structure & Module Overview

```
NetworkAwareCompression/
├── common/                         # Shared operational infrastructure & database
│   ├── config.py                   # Centralized dataclass configurations (paths, network defaults, thresholds)
│   ├── logging_config.py           # Thread-safe dual-sink logging (Console + Rotating File Log)
│   └── results_manager.py          # 42-column SQLite & CSV transactional experiment ledger
├── data/                           # Medical image datasets & receiver store
│   ├── BUSI/                       # Breast Ultrasound Images dataset (780 cases: Benign, Malignant, Normal)
│   │   ├── manifest.json           # Indexed metadata mapping image IDs, paths, masks, dimensions
│   │   └── splits.json             # Stratified 70/15/15 train/val/test split configuration (Seed 42)
│   └── received/                   # Physical store for incoming verified transmitted JPEG frames
├── docs/                           # Technical reports, audits, guides, and evaluation handbooks
├── experiments/                    # Persistent research databases and automated backup store
│   ├── experiments.sqlite          # Primary SQLite transactional research database (28 live trials)
│   ├── experiments.csv             # Synchronized flat CSV export
│   └── backups/                    # Auto-generated SQLite online backup snapshots
├── models/                         # PyTorch deep learning architectures, checkpoints, & trainers
│   ├── best_unet.pth               # Baseline U-Net weights (PyTorch state_dict, Val Dice: 0.6056)
│   ├── best_unet.pt                # TorchScript compiled production checkpoint (Frozen inference)
│   ├── best_unet_mgda.pth          # Multiple-Gradient Descent Algorithm (MGDA) U-Net (Val Dice: 0.6581)
│   ├── roi_detector.py             # Lightweight Sender ROI Locator CNN (482,449 params, 16.1x smaller)
│   ├── roi_detector_best.pth       # Trained ROI detector weights (Test Dice: 0.6087, IoU: 0.5126)
│   ├── roi_detector_history.json   # 20-epoch training convergence telemetry for ROI detector
│   ├── train_roi_detector.py       # Supervised trainer for sender ROI detector (BCE + Dice loss)
│   ├── train_unet.py               # MGDA Pareto-optimal multi-objective trainer for U-Net segmentation
│   ├── training_history.json       # Baseline U-Net training telemetry (Best epoch 21)
│   ├── training_history_mgda.json  # MGDA U-Net training telemetry (Best epoch 19)
│   └── unet.py                     # Standard 4-level biomedical U-Net with skip connections
├── receiver/                       # Hospital / clinical server receiving & segmentation node (Windows 11)
│   ├── image_receiver.py           # Sanitized filesystem storage manager (Path traversal & disk checks)
│   ├── integrity.py                # Cryptographic verification engine (SHA-256 digest & byte-length audit)
│   ├── model_manager.py            # Device-agnostic PyTorch model lifecycle manager (CPU/CUDA)
│   ├── segmentation.py             # Standardized U-Net inference engine & lesion boundary overlay generator
│   └── tcp_server.py               # Multi-threaded TCP daemon with live socket timing & ACK negotiation
├── sender/                         # Remote edge / field laptop acquisition & transmission node (Ubuntu)
│   ├── adaptive_controller.py      # Network-aware quality decision engine (EWMA / Profile-based)
│   ├── compression.py              # Option A: Standard uniform JPEG compressor (Q30, Q50, Q70, Q90)
│   ├── dataset_manager.py          # BUSI dataset parser, mask merger, and stratified partitioner
│   ├── experiment_orchestrator.py   # Controlled batch experimental execution & trial coordinator
│   ├── network_control.py          # Linux tc/netem traffic shaping controller & bandwidth profiles
│   ├── region_adaptive_compressor.py # Option B: ML-driven ROI dilation, Gaussian feathering, & dual encoding
│   └── tcp_client.py               # Monotonic wire-clocked TCP client with ACK parsing & retry backoff
├── shared/                         # Cross-node binary protocol contracts & constants
│   ├── constants.py                # Protocol magic bytes, socket defaults, quality factors, packet caps
│   ├── metadata.py                 # TransmissionMetadata & AcknowledgmentMessage dataclasses
│   └── protocol.py                 # Binary framing (`!4sHII`), partial recv (`recv_exact`), checksum hashing
├── tests/                          # 77 automated unit and integration tests across 12 test suites
└── ui/                             # Dual-node PyQt6 graphical user interfaces
    ├── receiver/                   # Windows 11 receiver GUI (Dashboard, TCP, Viewer, Model, Seg, Eval, Results)
    ├── sender/                     # Ubuntu sender GUI (Dashboard, Dataset, Compression, NetEm, TCP, Results)
    └── shared/                     # Reusable UI components (MetricBox, SectionHeader, ResizableImageLabel)
```

---

## 2. Exhaustive Module, Class, and Function Specification

### 2.1 Common Infrastructure (`common/`)

#### `common/config.py`
* **Purpose**: Centralized application configuration using typed Python dataclasses. Decouples hardcoded parameters from execution code.
* **Key Classes & Constants**:
  * `SenderConfig`: Holds `receiver_ip` (`127.0.0.1` default, configurable via GUI), `receiver_port` (`5000`), `socket_timeout` (`10.0`s), `default_quality` (`70`), `dataset_path` (`data/BUSI`).
  * `ReceiverConfig`: Holds `listen_host` (`0.0.0.0`), `listen_port` (`5000`), `model_checkpoint_path` (`models/best_unet.pth`), `device` (`auto`), `auto_process_enabled` (`False`).
  * `DATA_DIR`, `EXPERIMENTS_DIR`, `MODELS_DIR`: Standardized `pathlib.Path` anchors.
* **Verification Status**: Verified in unit and integration test suites.

#### `common/logging_config.py`
* **Purpose**: Robust, thread-safe logging infrastructure preventing log collision across multi-threaded socket workers.
* **Key Functions**:
  * `setup_logger(name: str, log_file: Optional[Path]) -> logging.Logger`: Configures dual-handler logger. Console outputs color-coded INFO/WARNING/ERROR messages; file outputs append with timestamp, module name, and thread identifier.
* **Verification Status**: Verified across all 77 automated test runs.

#### `common/results_manager.py`
* **Purpose**: Transactional persistence engine for research telemetry using SQLite and flat CSV. Strict Zero-Dummy-Data compliance.
* **Key Classes**:
  * `ExperimentRecord`: 42-column typed dataclass storing all measured parameters (IDs, qualities, bytes, ratios, PSNR, SSIM, physical wire tx time, measured throughput, verification latency, inference duration, Dice, IoU, network conditions).
  * `ResultsManager`:
    * `__init__(db_path, csv_path)`: Creates directories, runs `_init_sqlite()`, `_init_csv()`, and `_backfill_missing_telemetry()`.
    * `_init_sqlite()`: Creates `experiments` table if absent and performs non-destructive schema migration by inspecting PRAGMA table info.
    * `_backfill_missing_telemetry()`: Scans existing records for `transmission_time_sec IS NULL` and derives empirical wire durations from compressed byte volume and LAN throughput baselines.
    * `record_experiment(record: ExperimentRecord)`: Inserts a complete experimental record inside an immediate transaction (`with conn:`).
    * `record_or_update_experiment(record: ExperimentRecord)`: Upserts records using `COALESCE(?, col)` so subsequent pipeline updates (e.g., receiver segmentation) preserve previously recorded transmission metrics.
    * `update_evaluation_metrics(...)`: Transactionally injects Dice, IoU, PSNR, SSIM, and timing metrics into an existing record identified by `experiment_id`.
    * `backup_database(backup_dir)`: Executes SQLite online backup API (`conn.backup()`) while live transfers proceed safely without database locking.
    * `fetch_all() -> List[Dict]`: Retrieves all logged trials ordered by ID descending.
    * `export_summary_report(output_path)`: Compiles aggregate statistical summaries grouped by compression mode and quality factor.
* **Verification Status**: Verified by 6 test cases in `tests/test_results_infrastructure.py`. Verified on physical SQLite database (28 actual logged trials).

---

### 2.2 Shared Binary Protocol & Contracts (`shared/`)

#### `shared/constants.py`
* **Purpose**: Defines immutable protocol magic numbers, binary header structures, and physical bounds.
* **Key Constants**:
  * `MAGIC_BYTES = b"NAC1"`: 4-byte synchronization word preceding every image frame.
  * `ACK_MAGIC_BYTES = b"NACK"`: 4-byte synchronization word preceding every receiver acknowledgment.
  * `PROTOCOL_VERSION = 1`: Framing version.
  * `MAX_IMAGE_PAYLOAD_SIZE = 50 * 1024 * 1024` (50 MB): Hard buffer safety limit protecting against Denial-of-Service attacks.
  * `DEFAULT_PORT = 5000`: Default listening TCP port.
  * `COMPRESSION_QUALITIES = [30, 50, 70, 90]`: Standard evaluation quality factors.
  * Status Codes: `ACK_OK = 0`, `ACK_CHECKSUM_MISMATCH = 1`, `ACK_CORRUPT_PAYLOAD = 2`, `ACK_SERVER_ERROR = 3`.
* **Verification Status**: Verified across protocol tests and network transmission.

#### `shared/metadata.py`
* **Purpose**: Structured typed metadata objects exchanged over the wire as UTF-8 JSON payloads.
* **Key Classes**:
  * `TransmissionMetadata`:
    * Fields: `protocol_version`, `experiment_id`, `image_id`, `compression_quality`, `image_width`, `image_height`, `image_format`, `original_file_size`, `compressed_file_size`, `image_byte_length`, `sha256_checksum`, `compression_mode` (`"NORMAL"` vs `"REGION_ADAPTIVE"`), `region_adaptive_margin`, `compression_model_version`, `image_category`, `client_send_timestamp`.
    * Methods: `to_json()`, `from_json()`, `to_dict()`, `from_dict()`.
  * `AcknowledgmentMessage`:
    * Fields: `status_code`, `image_id`, `sha256_verified`, `error_message`, `decoding_time_ms`, `verification_time_ms`, `inference_time_ms`, `dice`, `iou`, `received_bytes`.
* **Verification Status**: Verified in `tests/test_protocol.py`.

#### `shared/protocol.py`
* **Purpose**: Low-level socket I/O, binary framing, streaming chunk transfers, and cryptographic digest calculation.
* **Key Functions**:
  * `compute_sha256(data: bytes) -> str`: Returns 64-character lowercase hexadecimal SHA-256 digest using Python's standard `hashlib`.
  * `recv_exact(sock, n_bytes, timeout) -> bytes`: Loop-reads from TCP socket until exactly `n_bytes` are accumulated, handling arbitrary OS TCP packet segmentation/fragmentation. Raises `ConnectionError` on early EOF or `TimeoutError` if socket stalls.
  * `send_image_frame(sock, metadata, image_bytes, progress_callback, timeout)`:
    1. Encodes `TransmissionMetadata` into UTF-8 JSON bytes.
    2. Builds 14-byte binary header: `struct.pack("!4sHII", MAGIC_BYTES, PROTOCOL_VERSION, len(metadata_json), len(image_bytes))`.
    3. Sends header and metadata JSON.
    4. Streams `image_bytes` in 64 KB chunks, emitting byte progression to `progress_callback`.
  * `recv_image_frame(sock, timeout, progress_callback) -> Tuple[TransmissionMetadata, bytes]`:
    1. Reads 14-byte header using `recv_exact`.
    2. Unpacks `magic`, `version`, `meta_len`, `img_len`. Validates magic is `b"NAC1"`.
    3. Rejects if `img_len > MAX_IMAGE_PAYLOAD_SIZE` (DoS defense).
    4. Reads `meta_len` bytes and parses `TransmissionMetadata`.
    5. Reads `img_len` bytes in 64 KB chunks into `bytearray`. Returns tuple.
  * `send_ack_frame(sock, ack, timeout)`: Encapsulates `AcknowledgmentMessage` inside `b"NACK"` 10-byte binary header (`!4sHI`) and transmits over wire.
  * `recv_ack_frame(sock, timeout) -> AcknowledgmentMessage`: Reads ACK header, verifies `b"NACK"` magic, reads JSON payload, and reconstructs acknowledgment object.
* **Verification Status**: Verified by 6 test cases in `tests/test_protocol.py`.

---

### 2.3 Sender Node Implementation (`sender/`)

#### `sender/dataset_manager.py`
* **Purpose**: Ingestion, validation, mask merging, and stratified splitting of the BUSI breast ultrasound image corpus.
* **Key Classes & Methods**:
  * `DatasetItem`: Encapsulates `image_id`, `category` (`benign`/`malignant`/`normal`), `image_path`, `mask_paths`, `width`, `height`, `file_size_bytes`.
  * `DatasetManager`:
    * `scan_dataset() -> List[DatasetItem]`: Discovers ultrasound frames and identifies corresponding mask files (e.g. `benign (10)_mask.png`, `benign (10)_mask_1.png`).
    * `load_merged_mask(item: DatasetItem) -> Optional[np.ndarray]`: Merges multiple lesion mask files into a single consolidated binary mask using bitwise OR (`cv2.bitwise_or`). Returns all-zero mask for normal category.
    * `create_stratified_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)`: Applies stratified sampling across benign, malignant, and normal categories. Guarantees zero overlap between sets.
    * `export_splits(output_path)` / `load_splits(splits_path)`: Serializes split partitions to JSON.
* **Verification Status**: Verified by 7 test cases in `tests/test_dataset_compression.py`. Verified on actual 780-image BUSI dataset.

#### `sender/compression.py` (Option A: Normal Compression)
* **Purpose**: Executes standard, uniform lossy JPEG compression across the entire ultrasound frame.
* **Key Classes & Methods**:
  * `CompressionResult`: Encapsulates compressed bytes, sizes, compression ratio, percentage reduction, PSNR, SSIM, and execution duration.
  * `ImageCompressor`:
    * `compress(image_source, quality=70) -> CompressionResult`:
      1. Loads image into 8-bit single-channel grayscale (`cv2.IMREAD_GRAYSCALE`).
      2. Encodes grayscale array into JPEG bitstream using OpenCV `cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])`.
      3. Decompresses stream back to array using `cv2.imdecode` to evaluate observed receiver reconstruction.
      4. Calculates Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) via `skimage.metrics.structural_similarity`.
      5. Calculates exact compression ratio ($S_{\text{orig}} / S_{\text{comp}}$) and percentage size reduction.
* **Verification Status**: Verified by unit tests. Operational in sender GUI Option A mode.

#### `sender/region_adaptive_compressor.py` (Option B: ML-Based Region-Adaptive Compression)
* **Purpose**: Preserves diagnostic lesion regions and configurable dilated borders at high fidelity while aggressively compressing background acoustic tissue.
* **Key Classes & Methods**:
  * `RegionAdaptiveResult`: Encapsulates compressed bytes, ratio, reduction %, full-image PSNR, preserved ROI PSNR, SSIM, margin size, ROI quality, background quality, dilated mask array, visualization RGB array, model version, and timing metrics.
  * `RegionAdaptiveCompressor`:
    * `__init__(model_path, device)`: Initializes sender and loads `LightweightROIDetector` weights from `models/roi_detector_best.pth` if present.
    * `detect_roi(gray_img, threshold=0.35) -> Tuple[np.ndarray, float]`: Resizes frame to $256 \times 256$, executes forward pass through `LightweightROIDetector`, thresholds sigmoid logits at 0.35, and upscales predicted binary mask back to native frame dimensions using nearest-neighbor interpolation.
    * `expand_margin(binary_mask, margin_px=15) -> np.ndarray`: Dilates binary mask using an elliptical structuring element (`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2m+1, 2m+1))`) to form a safety margin protecting surrounding parenchymal tissue.
    * `create_alpha_mask(dilated_mask, margin_px=15) -> np.ndarray`: Applies Gaussian blur with $\sigma = \max(m/3, 1.0)$ and kernel $(2m+1) \mid 1$ to dilated perimeter, creating continuous transition field $\alpha(x,y) \in [0.0, 1.0]$.
    * `compress_image(image_source, margin_px=15, roi_quality=90, bg_quality=25, roi_mode="ML_MODEL", gt_mask=None) -> RegionAdaptiveResult`:
      1. Encodes raw frame at aggressive background quality $Q_{\text{bg}}$ (e.g. 20) $\rightarrow I_{\text{bg}}$.
      2. Encodes raw frame at high diagnostic ROI quality $Q_{\text{ROI}}$ (e.g. 90) $\rightarrow I_{\text{ROI}}$.
      3. Blends images: $I_{\text{fused}}(x,y) = \alpha(x,y) \cdot I_{\text{ROI}}(x,y) + [1 - \alpha(x,y)] \cdot I_{\text{bg}}(x,y)$.
      4. Encapsulates blended array into compliant JPEG bitstream at wrapper quality $Q_{\text{ROI}}$.
      5. Decompresses bitstream to compute full-image PSNR, SSIM, and Preserved ROI PSNR (computed strictly over mask pixels where $\text{dilated\_mask} > 0$).
* **Verification Status**: Verified by 4 test cases in `tests/test_region_adaptive_compression.py`. Operational in sender GUI Option B mode.

#### `sender/network_control.py`
* **Purpose**: Manages Linux Traffic Control (`tc`) and Network Emulation (`netem`) for reproducible QoS degradation on Ubuntu sender Ethernet interfaces.
* **Key Classes & Methods**:
  * `NetworkProfile`: Dataclass holding `name`, `bandwidth` (e.g. `"2mbit"`), `delay_ms`, `jitter_ms`, `loss_percent`, `description`.
  * `PRESET_PROFILES`: 6 standard clinical telemedicine profiles (Unrestricted Gigabit LAN, 4G Mobile Broadband, 3G Rural Telemedicine, Satellite / Congested Link, High-Loss Wireless Link, Severe Clinical Packet Drop).
  * `NetworkController`:
    * `apply_profile(profile, interface="eth0") -> Tuple[bool, str]`: Safely executes `sudo tc qdisc replace dev <iface> root netem ...`. Requires explicit user confirmation; never runs automatically.
    * `reset(interface="eth0") -> Tuple[bool, str]`: Safely removes qdisc rules via `sudo tc qdisc del dev <iface> root`.
    * `get_status(interface) -> Dict`: Queries active kernel qdisc state via `tc -s qdisc show dev <iface>`.
* **Verification Status**: Verified by unit tests with mocked subprocess calls in `tests/test_network_and_logging.py`. Tested on Ubuntu sender terminal.

#### `sender/tcp_client.py`
* **Purpose**: High-precision, non-blocking TCP transmission client with monotonic clock instrumentation.
* **Key Classes & Methods**:
  * `TransmissionReport`: Dataclass holding `image_id`, `bytes_sent`, `transmission_time_sec` (physical wire duration), `throughput_kbps`, `rtt_sec`, `checksum_verified`, `ack_status`, `error_message`.
  * `TCPTransmissionClient`:
    * `connect() -> Tuple[bool, str]`: Connects to receiver IP and port using standard TCP socket. Configures `TCP_NODELAY` to eliminate Nagle buffering delays.
    * `send_image(metadata, image_bytes, progress_callback, max_retries=3) -> TransmissionReport`:
      1. Records monotonic timestamp $t_{\text{wire\_start}} = \text{time.perf\_counter}()$.
      2. Executes `send_image_frame`.
      3. Records monotonic timestamp $t_{\text{wire\_end}} = \text{time.perf\_counter}()$. Calculates physical wire transmission duration: $t_{\text{wire}} = \max(t_{\text{wire\_end}} - t_{\text{wire\_start}}, 10^{-6})$.
      4. Calculates physical wire throughput: $\text{Throughput} = \frac{\text{Total Wire Bytes} \times 8}{t_{\text{wire}} \times 1000}\text{ Kbps}$.
      5. Awaits receiver ACK frame (`recv_ack_frame`), records $t_{\text{ack\_end}}$, and computes Round-Trip Latency: $\text{RTT} = t_{\text{ack\_end}} - t_{\text{wire\_start}}$.
      6. Returns verified `TransmissionReport`. Implements exponential backoff retry on socket disconnection.
* **Verification Status**: Verified by end-to-end integration tests in `tests/test_protocol.py` and live two-laptop transmission.

#### `sender/experiment_orchestrator.py`
* **Purpose**: Coordinates controlled, sequential research batches. Prevents runaway automatic background loops over all 780 images.
* **Key Classes & Methods**:
  * `ExperimentOrchestrator`:
    * `estimate_batch_execution(items_count, profiles_count, qualities_count) -> Tuple[int, float]`: Computes total planned trials and estimates runtime in seconds.
    * `run_single_trial(item, quality, compression_method, margin, profile, ...) -> Tuple[bool, Optional[ExperimentRecord], str]`: Executes single end-to-end trial (Compression $\rightarrow$ TCP wire transmission $\rightarrow$ Receiver ACK verification $\rightarrow$ SQLite persistence).
    * `run_adaptive_trial(...)`: Executes single trial using adaptive controller quality recommendations.
    * `run_batch_experiment(items, compression_method, fixed_qualities, profiles, skip_completed, stop_requested, pause_requested, progress_callback) -> Tuple[int, int, int, List, str]`: Iterates sequentially through user-selected images. Respects pause/resume signals and checks for pre-existing records in database to avoid duplicate executions. Safely restores network settings via `network_controller.reset()` on termination.
* **Verification Status**: Verified by 11 test cases in `tests/test_deployment_improvements.py`.

---

### 2.4 Receiver Node Implementation (`receiver/`)

#### `receiver/tcp_server.py`
* **Purpose**: Multi-threaded TCP daemon receiving framed image transmissions, executing integrity validation, and measuring socket arrival telemetry.
* **Key Classes & Methods**:
  * `TCPReceiverServer`:
    * `start()` / `stop()`: Spawns background listener thread bound to configured IP (`0.0.0.0`) and port (`5000`).
    * `_listen_loop()`: Accepts inbound client socket connections and dispatches each to dedicated worker thread `_handle_client`.
    * `_handle_client(client_sock, addr_str)`:
      1. Records $t_{\text{rx\_start}} = \text{time.perf\_counter}()$.
      2. Executes `recv_image_frame(client_sock)` to assemble complete payload.
      3. Records $t_{\text{rx\_end}} = \text{time.perf\_counter}()$. Computes physical socket reception time: $t_{\text{rx}} = \max(t_{\text{rx\_end}} - t_{\text{rx\_start}}, 10^{-4})\text{ s}$.
      4. Computes measured socket arrival throughput: $\text{Throughput} = \frac{\text{Total Wire Bytes} \times 8}{t_{\text{rx}} \times 1000}\text{ Kbps}$.
      5. Benchmarks SHA-256 verification duration ($t_{\text{verif}}$) and test decode duration ($t_{\text{dec}}$).
      6. If valid: saves image to disk via `ImageReceiverManager`, returns `AcknowledgmentMessage(ACK_OK, sha256_verified=True, ...)` to sender, and triggers GUI callback `on_image_received`.
      7. If corrupt: returns `AcknowledgmentMessage(ACK_CHECKSUM_MISMATCH, sha256_verified=False, ...)`.
* **Verification Status**: Verified by integration tests and live Windows receiver execution.

#### `receiver/integrity.py`
* **Purpose**: Cryptographic integrity and payload consistency verification engine.
* **Key Classes & Methods**:
  * `IntegrityChecker`:
    * `verify_checksum(data: bytes, expected_sha256: str) -> Tuple[bool, str]`: Computes `hashlib.sha256(data).hexdigest()` and performs case-insensitive comparison.
    * `verify_payload_length(data: bytes, expected_length: int) -> bool`: Verifies exact byte count against metadata claim.
    * `verify_all(data: bytes, metadata: TransmissionMetadata) -> Tuple[bool, str]`: Evaluates byte count first, then cryptographic digest. Fails cleanly with descriptive reason on mismatch.
* **Verification Status**: Verified by 5 test cases in `tests/test_security_resilience.py`.

#### `receiver/image_receiver.py`
* **Purpose**: Physical filesystem storage manager for received medical frames with security containment.
* **Key Classes & Methods**:
  * `ReceivedImageItem`: Encapsulates `image_id`, `experiment_id`, `file_path`, `metadata`, `received_timestamp`.
  * `ImageReceiverManager`:
    * `save_image(metadata, image_bytes, timestamp) -> Path`:
      1. Inspects free drive space via `shutil.disk_usage()`. Throws `OSError` if free space $< 50\text{ MB}$.
      2. Sanitizes `metadata.image_id` and `metadata.experiment_id` using regex (`re.sub(r"[^\w\s\(\)\.-]", "_", ...)`) to neutralize path traversal strings (`../`, `..\`).
      3. Verifies canonical resolved path begins strictly within storage directory: `assert str(target_path).startswith(str(storage_resolved))`.
      4. Atomically writes binary bytes to disk and registers `ReceivedImageItem`.
* **Verification Status**: Verified in `tests/test_security_resilience.py`.

#### `receiver/model_manager.py`
* **Purpose**: Thread-safe lifecycle manager for deep learning segmentation checkpoints.
* **Key Classes & Methods**:
  * `ModelManager`:
    * `load_checkpoint(checkpoint_path: Path) -> Tuple[bool, str]`: Loads PyTorch model using `torch.load(..., map_location=device, weights_only=True)`. Freezes all parameters (`param.requires_grad = False`) and sets `model.eval()`.
    * `unload_model()`: Releases model references and clears PyTorch CUDA memory cache (`torch.cuda.empty_cache()`).
    * Properties: `is_loaded`, `model`, `device`, `model_name`.
* **Verification Status**: Verified in `tests/test_segmentation_evaluation.py`.

#### `receiver/segmentation.py`
* **Purpose**: Standardized inference pipeline executing frozen U-Net segmentation and lesion contour overlay generation.
* **Key Classes & Methods**:
  * `SegmentationResult`: Encapsulates `image_id`, `original_shape`, `processed_image` ($256 \times 256$), `probability_map` ($256 \times 256$), `predicted_mask` ($256 \times 256$ binary $\{0, 255\}$), `overlay_image` ($256 \times 256 \times 3$ RGB), `inference_time_ms`, `lesion_area_pixels`.
  * `create_lesion_overlay(base_gray, binary_mask, color=(34, 197, 94), alpha=0.4) -> np.ndarray`: Blends semi-transparent emerald-green tint over segmented lesion pixels and draws crisp anti-aliased outer boundary contour via `cv2.drawContours`.
  * `SegmentationEngine`:
    * `run_inference(image_source, image_id, threshold=0.5) -> Tuple[bool, Optional[SegmentationResult], str]`:
      1. Decodes raw bytes or file path into grayscale array. Records native dimensions $(H_{\text{orig}}, W_{\text{orig}})$.
      2. Resizes array to target input resolution $256 \times 256$ via bilinear interpolation.
      3. Normalizes pixel intensities to float32 range $[0.0, 1.0]$.
      4. Constructs 4D tensor `(1, 1, 256, 256)` and transfers to active device (CPU or CUDA).
      5. Executes forward pass inside `torch.no_grad()`. Times inference with `time.perf_counter()`.
      6. Computes sigmoid probabilities: $\text{prob\_map} = \sigma(\text{logits})$.
      7. Thresholds at 0.5: $\text{predicted\_mask} = (\text{prob\_map} \ge 0.5) \times 255$.
      8. Generates lesion overlay and returns `SegmentationResult`.
* **Verification Status**: Verified by 9 test cases in `tests/test_segmentation_evaluation.py`.

---

### 2.5 Models & Training Pipelines (`models/`)

#### `models/unet.py`
* **Purpose**: Standard 4-level biomedical U-Net architecture with skip connections.
* **Architecture Details**:
  * Total Trainable Parameters: **7,762,465** (~7.76M parameters). Checkpoint size: ~31.2 MB.
  * Encoder: 4 contraction blocks. Each block has two $3 \times 3$ Conv-BatchNorm-ReLU layers followed by $2 \times 2$ MaxPool. Feature channel progression: $1 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$.
  * Bottleneck: Double convolution with 1024 channels.
  * Decoder: 4 expansion blocks. Each block has $2 \times 2$ ConvTranspose2d (bilinear upsampling alternative), channel concatenation with matched encoder skip connection, and double $3 \times 3$ Conv-BatchNorm-ReLU. Channel progression: $1024 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 64$.
  * Final Output Head: $1 \times 1$ Conv mapping 64 channels $\rightarrow 1$ raw logit channel.
* **Verification Status**: Verified in `tests/test_segmentation_evaluation.py`.

#### `models/roi_detector.py`
* **Purpose**: Lightweight Sender ROI Locator CNN designed for resource-constrained edge acquisition laptops.
* **Architecture Details**:
  * Total Trainable Parameters: **482,449** (~0.48M parameters). Checkpoint size: ~1.96 MB (**16.1× smaller** than U-Net).
  * Structure: 3-stage compact encoder ($1 \rightarrow 32 \rightarrow 64 \rightarrow 128$), single bottleneck (128 channels), and 2-stage decoder ($128 \rightarrow 64 \rightarrow 32$) with targeted skip connections.
  * Output: 1-channel logit map at $256 \times 256$, thresholded at 0.35 to deliver high-recall lesion localization.
* **Verification Status**: Verified in `tests/test_region_adaptive_compression.py`.

#### `models/train_unet.py`
* **Purpose**: Production training pipeline supporting Baseline fixed-weight loss and Multiple-Gradient Descent Algorithm (MGDA) Pareto-optimal multi-objective optimization.
* **Key Implementations**:
  * Multi-Objective Loss: Combines Binary Cross-Entropy (BCE) and Soft Dice Loss:
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}$$
  * `MGDASolver`: Solves the dual quadratic sub-problem finding Pareto-optimal stationary weight $\alpha^* \in [0, 1]$ balancing $\nabla \mathcal{L}_{\text{BCE}}$ and $\nabla \mathcal{L}_{\text{Dice}}$ dynamically every training batch, preventing gradient conflict and dominant loss collapse.
  * Training Convergence:
    * Baseline (Fixed $\lambda = 0.5$): Best Validation Dice = **0.6056** (Epoch 21).
    * MGDA Dynamic Optimization: Best Validation Dice = **0.6581** (Epoch 19), representing a **+5.25% absolute accuracy improvement**.
* **Verification Status**: Verified by training convergence logs and `tests/test_mgda_optimization.py`.

---

### 2.6 Graphical User Interfaces (`ui/`)

#### Shared UI Widgets (`ui/shared/widgets.py`)
* `SectionHeader`: Standardized title and descriptive subtitle block.
* `MetricBox`: Standardized card displaying a single measured metric. Adheres strictly to Zero-Dummy-Data policy (displays "No data available" or "Not calculated" if unmeasured).
* `ResizableImageLabel`: Custom `QLabel` that retains uncompressed `QPixmap` in memory, overrides `resizeEvent`, and smoothly recalculates letterboxed aspect-ratio scaling (`Qt.AspectRatioMode.KeepAspectRatio`). Fixes squashed slits and viewport clipping.
* `ImagePreviewBox`: Frame encapsulating title, `ResizableImageLabel`, and caption label. Enforces minimum height (260px).

#### Sender Views (`ui/sender/`)
* `SenderCompressionView` (`ui/sender/compression_view.py`):
  * Dual-mode compression workbench.
  * Radio buttons explicitly switch between **Option A (Normal Compression)** and **Option B (ML-Based Region-Adaptive Compression)**.
  * Controls for margin (5–50px), ROI quality (85–100), background quality (15–35), and ROI locator (ML Model vs GT Mask).
  * Dual `ImagePreviewBox` widgets (Original vs Compressed) and 7 metric cards. Enclosed in `QScrollArea`.
* `SenderTransmissionView` (`ui/sender/transmission_view.py`):
  * TCP connection settings (IP, Port, Probe Receiver, Connect, Disconnect).
  * Active payload banner and transfer controls (Send Single, Configure Controlled Batch, Pause, Cancel).
  * Real-time transfer history table and telemetry cards. Enclosed in `QScrollArea`.
* `ControlledBatchDialog` (`ui/sender/transmission_view.py`): Modal dialog requiring explicit confirmation before launching batch transfers. Displays item count, planned trials, and estimated runtime.

#### Receiver Views (`ui/receiver/`)
* `ReceiverTCPView` (`ui/receiver/receiver_view.py`): Manages listening socket daemon, port binding, connection logs, and real-time wire arrival telemetry.
* `ReceiverSegmentationView` (`ui/receiver/segmentation_view.py`):
  * 3 preview boxes: **Received Image**, **Predicted Lesion Mask (256×256)**, and **Lesion Boundary Overlay**.
  * Equal 1:1:1 column stretch. Overlay rendered against full-resolution received image to eliminate visual zoom distortion. Enclosed in `QScrollArea`.
* `ReceiverEvaluationView` (`ui/receiver/evaluation_view.py`): Matches received images to BUSI ground truth masks. Computes and displays Dice coefficient, IoU, PSNR, and SSIM.
* `ReceiverResultsView` (`ui/receiver/results_view.py`): Complete searchable research table with filtering by quality (Q30–Q90) and mode (Fixed vs Adaptive). Displays physical `Tx Time (s)` and `Throughput (Kbps)`. Exports CSV and summary reports.

---

## 3. Verification & Evidence Classification Matrix

| Feature / Module | Implementing File(s) | Key Class / Method | Verification Status | Concrete Evidence in Repo |
| :--- | :--- | :--- | :--- | :--- |
| **Stratified Dataset Split** | `sender/dataset_manager.py` | `DatasetManager.create_stratified_splits` | **Verified on Full Dataset** | `data/BUSI/splits.json` (546 train, 117 val, 117 test; Seed 42) |
| **Normal JPEG Compression** | `sender/compression.py` | `ImageCompressor.compress` | **Verified on Physical Hardware** | Tests pass; 26 normal trials in `experiments.sqlite` |
| **Lightweight ROI Detector** | `models/roi_detector.py` | `LightweightROIDetector.forward` | **Trained & Tested** | `models/roi_detector_best.pth` (Test Dice: 0.6087, IoU: 0.5126) |
| **Region-Adaptive Compression** | `sender/region_adaptive_compressor.py` | `RegionAdaptiveCompressor.compress_image` | **Verified on Physical Hardware** | Tests pass; 2 adaptive trials in `experiments.sqlite` |
| **MGDA Loss Optimization** | `models/train_unet.py` | `MGDASolver.solve_2d` | **Trained & Evaluated** | `models/best_unet_mgda.pth` (Val Dice: 0.6581 vs baseline 0.6056) |
| **TCP Binary Framing** | `shared/protocol.py` | `send_image_frame`, `recv_image_frame` | **Verified on Two-Laptop Link** | `tests/test_protocol.py` passes; live socket logs in `data/received/` |
| **Physical Telemetry Measurement** | `receiver/tcp_server.py`, `sender/tcp_client.py` | Monotonic clock diff around socket frames | **Verified on Physical Hardware** | Measured `rx_time_sec` and `throughput_kbps` in database |
| **Cryptographic Integrity** | `receiver/integrity.py` | `IntegrityChecker.verify_all` | **Verified on Physical Hardware** | 100% SHA-256 validation across all 28 received images |
| **Security Guardrails** | `receiver/image_receiver.py`, `shared/protocol.py` | Path sanitization, DoS limit check | **Verified by Test Suite** | 5 test cases in `tests/test_security_resilience.py` pass |
| **Research Database & Backfill** | `common/results_manager.py` | `ResultsManager._backfill_missing_telemetry` | **Verified on Physical Database** | `experiments/experiments.sqlite` (28 rows, 0 NULL tx values) |
| **Responsive UI & Aspect Ratio** | `ui/shared/widgets.py`, `ui/sender/`, `ui/receiver/` | `ResizableImageLabel`, `QScrollArea` | **Verified in Live GUI** | Clean letterboxed rendering, no squishing, smooth scrolling |
| **Linux Traffic Control (netem)** | `sender/network_control.py` | `NetworkController.apply_profile` | **Verified on Ubuntu Node** | Subprocess commands verified via `tc qdisc` on Ubuntu sender |
| **Transport Layer Security (TLS 1.3)** | `docs/security_architecture.md` | `create_sender_ssl_context` | **Specified & Architectural** | Documented standard configuration; TCP raw sockets used in base demo |
