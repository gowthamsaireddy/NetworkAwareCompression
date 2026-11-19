# Comprehensive Project Evaluation & Viva Voce Defense Dossier (2-Hour Defense Master Guide)

## Project Title: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

---

# MODULE 1: PROBLEM FORMULATION & CORE RESEARCH THESIS

### 1. Fundamentals
In modern healthcare systems, ultrasound imaging is the primary frontline modality for screening and diagnosing breast lesions (differentiating between benign fibroadenomas and malignant carcinomas). However, access to experienced radiologists and high-performance diagnostic workstations is disproportionately concentrated in tertiary urban hospitals. Rural clinics, community healthcare centers, and mobile emergency units lack on-site radiological expertise.

Tele-ultrasound seeks to bridge this divide by acquiring scans in remote field settings and streaming them over wireless or satellite networks to centralized hospitals for diagnostic triage and automated Computer-Aided Diagnosis (CAD) via deep learning models.

However, streaming ultrasound images across real-world communication links introduces a fundamental engineering tradeoff:
1. **The Network Bottleneck**: Uncompressed DICOM or lossless PNG ultrasound images range from hundreds of kilobytes to several megabytes per frame. Over constrained cellular (3G/4G), microwave, or satellite links, transmitting raw scans causes severe latency, transmission buffer bloat, and socket timeouts.
2. **The Clinical AI Dilemma**: Applying lossy image compression (such as JPEG) reduces the transmission payload and transmission time by 80% to 95%. However, lossy compression discards high-frequency spatial details and introduces compression artifacts (e.g., blocking boundaries, edge blur, ringing). Because deep learning segmentation networks (like U-Net) rely heavily on subtle acoustic textures, speckle patterns, and lesion boundary gradients, aggressive compression can severely degrade automated lesion boundary delineation, risking misdiagnosis.

### 2. Why We Use This in This Project
This project addresses this exact clinical and engineering bottleneck. Instead of treating image compression, network transmission, and deep learning segmentation as isolated silos, our research integrates them into an end-to-end, closed-loop, network-aware system. The core thesis is:
> *By dynamically measuring real-time network throughput and adapting JPEG compression quality to match instantaneous channel capacity, we can minimize transmission latency across degraded channels while preserving the diagnostic segmentation accuracy of downstream clinical AI models.*

### 3. Exact Repository Implementation
- Problem scope and requirements: `README.md` (Lines 1-85)
- Telemetry & coordination: `sender/experiment_orchestrator.py` (Lines 1-320)
- End-to-end evaluation: `experiments/evaluate_baseline.py` (Lines 1-210)
- Configurable thresholds: `sender/adaptive_controller.py` (Lines 20-37)

### 4. Internal Mechanism & System Workflow
1. The **Ubuntu Sender** monitors channel conditions (or receives netem-emulated link parameters).
2. The **Adaptive Controller** tracks physical transmission throughput using an Exponentially Weighted Moving Average (EWMA) and selects the highest viable JPEG quality factor $Q \in \{30, 50, 70, 90\}$ that fits the current link bandwidth.
3. The **Compression Engine** encodes the raw ultrasound scan to the target quality factor in-memory, computes mathematical distortion metrics (PSNR, SSIM), and generates a cryptographic SHA-256 digest of the compressed payload.
4. The **TCP Client** streams the image across a physical Ethernet connection using a custom binary wire framing protocol (`NAC1`).
5. The **Windows Receiver** ingests the raw socket stream, verifies payload length and SHA-256 integrity, persists the scan to storage, and passes the reconstructed image into a **frozen U-Net** segmentation model.
6. The model infers the binary lesion mask, which is evaluated against pixel-perfect ground-truth annotations to compute the Dice Similarity Coefficient (DSC) and Intersection over Union (IoU).
7. All telemetry (compression size, PSNR, SSIM, theoretical delay, measured wire time, throughput, Dice, IoU) is logged to SQLite/CSV databases and displayed live on desktop GUIs.

### 5. Input, Processing, and Output
- **Input**: Raw Breast Ultrasound Images (BUSI) from disk (grayscale or RGB, variable dimensions up to $1000 \times 800$).
- **Processing**: Channel estimation $\rightarrow$ Dynamic JPEG quantization $\rightarrow$ SHA-256 hashing $\rightarrow$ TCP wire streaming $\rightarrow$ Checksum validation $\rightarrow$ Tensor preprocessing ($256 \times 256$, normalized $[0, 1]$) $\rightarrow$ Frozen U-Net forward pass $\rightarrow$ Metric calculation.
- **Output**: Transmitted image file, predicted segmentation mask, comparative RGB overlay, CSV/SQLite experiment records, and live GUI visual feedback.

### 6. Module Interconnections
The problem statement governs the entire directory tree:
- `sender/` owns the upstream transmission, compression, and network shaping.
- `shared/` governs wire protocol standards, framing structures, and cryptographic integrity.
- `receiver/` owns socket reception, storage, AI model execution, and clinical evaluation.
- `common/` provides persistent database logging, configuration management, and file routing.

### 7. Mathematical Foundation
The trade-off is formalized as a multi-objective optimization problem:
$$\min_{Q \in \{30, 50, 70, 90\}} \left[ \lambda \cdot T_{\text{tx}}(S(Q), B) + (1 - \lambda) \cdot \mathcal{L}_{\text{seg}}(M(I_Q), M(I_{\text{orig}})) \right]$$
Where:
- $S(Q)$ is the compressed image byte size at quality $Q$.
- $B$ is the channel bandwidth in bits per second.
- $T_{\text{tx}} = \frac{S(Q) \times 8}{B}$ is the transmission delay.
- $\mathcal{L}_{\text{seg}}$ is the segmentation degradation loss, defined as:
  $$\Delta \text{Dice} = \text{Dice}(M(I_{\text{orig}}), G) - \text{Dice}(M(I_Q), G)$$
- $\lambda \in [0, 1]$ represents the operational priority weighting between latency and segmentation accuracy.

### 8. Concrete Project Data & Examples
- **Uncompressed baseline**: Baseline ultrasound images average ~150 KB to 450 KB in PNG format. Over a 512 Kbps rural satellite link, transmitting an uncompressed 300 KB image takes $\approx 4.8$ seconds.
- **Compressed at Q30**: Compresses the image down to $\approx 18$ KB (a 94% size reduction). Transmission takes only $\approx 0.28$ seconds.
- **Clinical AI Impact**: On our trained U-Net evaluated on 117 held-out test scans, uncompressed baseline mean Dice is $0.6552$. When compressed at Q90 or Q70, Dice remains highly preserved ($> 0.64$). Even at Q50, the lesion boundary is largely intact. At Q30, high-frequency boundary gradients blur, which can reduce boundary precision on subtle malignant lesions.

### 9. How to State This in the Viva
> *"Respected Examiners, the primary objective of this project is solving the clinical-telemedicine dilemma: How can we transmit ultrasound scans over constrained emergency and rural networks with minimal latency, without corrupting the diagnostic accuracy of deep learning models used for lesion detection? To answer this, we built a physical dual-laptop testbed connecting an Ubuntu sender running Linux kernel traffic control to a Windows receiver running an NVIDIA GPU-accelerated frozen U-Net model. Our system dynamically adapts JPEG compression based on real-time channel bandwidth, achieving fast transmission while maintaining clinically acceptable segmentation performance."*

### 10. Viva Questions & Model Answers
- **Q1: Why did you choose ultrasound instead of X-ray or CT scans?**
  *Answer*: Ultrasound is the most portable, low-cost, and non-ionizing screening modality available for rural and disaster-response telemedicine (e.g., handheld point-of-care ultrasound, POCUS). However, ultrasound suffers from significant acoustic speckle noise and low contrast, making deep learning models especially sensitive to compression artifacts compared to high-contrast CT or X-ray scans.
- **Q2: Why not just use lossless compression like PNG or JPEG-LS?**
  *Answer*: Lossless compression on ultrasound images yields very modest compression ratios (typically between 1.2:1 and 2:1) due to random acoustic speckle patterns that lack compressible mathematical redundancies. In severely constrained networks (128 Kbps to 512 Kbps), lossless compression is insufficient to meet clinical latency requirements. Lossy compression achieves ratios from 8:1 to 25:1, which is necessary for low-bandwidth telemedicine.

---

# MODULE 2: DUAL-LAPTOP PHYSICAL TESTBED & HARDWARE TOPOLOGY

### 1. Fundamentals
In computer networking research, testing transmission protocols and rate-adaptation algorithms requires an environment that exhibits real physical network phenomena: socket buffering, physical Network Interface Card (NIC) serialization, kernel packet queuing, TCP sliding windows, and round-trip acknowledgments.

### 2. Why We Use This in This Project
Many academic papers simulate networks purely in memory using `time.sleep()` or synthetic random delays. This simulation approach is methodologically flawed:
- In-memory simulation bypasses the operating system's network stack entirely.
- It fails to capture physical TCP behavior: slow-start, packet retransmissions upon packet loss, TCP ACK delays, and receiver buffer exhaustion.
- It ignores the real serialization and deserialization overhead of network sockets.

To ensure our experimental findings have genuine empirical validity, we deployed a **physical dual-laptop testbed** connected via a dedicated Ethernet link.

### 3. Exact Repository Implementation
- Deployment architectures: `docs/architecture.md` (Lines 1-135)
- Ubuntu sender launch script: `run_sender.sh` (Lines 1-45)
- Windows receiver launch script: `run_receiver.bat` (Lines 1-35)
- Physical socket client: `sender/tcp_client.py` (Lines 20-110)
- Multi-threaded socket daemon: `receiver/tcp_server.py` (Lines 28-98)

### 4. Internal Mechanism & System Workflow
```
+-------------------------------------------------------------+
|                     UBUNTU LAPTOP (SENDER)                  |
| - OS: Ubuntu Linux                                          |
| - Kernel Subsystem: iproute2 / tc / netem                   |
| - Role: Dataset Provider, JPEG Compression, Traffic Control |
| - TCP Client: socket.connect((RECEIVER_IP, PORT))           |
+-------------------------------------------------------------+
                              │
                    [Physical Ethernet Link]
                   CAT5e/6 Direct Cable / LAN
                              │
+-------------------------------------------------------------+
|                    WINDOWS LAPTOP (RECEIVER)                |
| - OS: Windows 11 Physical Machine                           |
| - Hardware Accelerator: NVIDIA GeForce RTX 3050 6GB Laptop  |
| - Framework: PyTorch 2.6.0+cu124                            |
| - Role: Socket Daemon, Checksum Verifier, Frozen U-Net      |
+-------------------------------------------------------------+
```
1. **Ubuntu Linux (Sender Node)**:
   - Linux is selected for the sender because the Linux kernel natively includes `tc` (Traffic Control) and `netem` (Network Emulator) in its `iproute2` subsystem.
   - This allows kernel-level queuing discipline (`qdisc`) manipulation directly on the physical network interface (e.g., `eth0`), deterministically throttling rate and injecting jitter and packet loss.
2. **Windows 11 (Receiver Node)**:
   - Represents the central hospital diagnostic workstation.
   - Houses the dedicated NVIDIA GeForce RTX 3050 GPU running PyTorch with CUDA 12.4 for real-time deep learning inference.
   - Runs a multi-threaded TCP server listening on port 5000.

### 5. Input, Processing, and Output
- **Input**: Target IP address (e.g., `192.168.1.100` or local testbed IP) and port (default `5000`).
- **Processing**: Pre-connection TCP handshake $\rightarrow$ Socket option setup (`SO_REUSEADDR`, `TCP_NODELAY`, custom timeouts) $\rightarrow$ Full-duplex byte stream transmission over physical Ethernet.
- **Output**: Established physical TCP connection, transmission timing measured via high-resolution hardware monotonic clocks (`time.perf_counter()`).

### 6. Module Interconnections
- `sender/network_control.py` configures the Ubuntu kernel qdisc on the physical NIC.
- `sender/tcp_client.py` initiates the socket connection across the wire.
- `receiver/tcp_server.py` accepts the socket, spawns a handling thread, and delegates byte reception to `shared/protocol.py`.

### 7. Mathematical Foundation
Theoretical transmission delay across the physical wire is given by:
$$T_{\text{theoretical}} = \frac{L_{\text{payload}} \times 8}{B_{\text{configured}}}$$
Where $L_{\text{payload}}$ is the total bytes sent (Header + Metadata + Image bytes), and $B_{\text{configured}}$ is the throttled bandwidth in bits per second.

The actual measured transmission time $T_{\text{measured}}$ includes:
$$T_{\text{measured}} = T_{\text{serialization}} + T_{\text{propagation}} + T_{\text{queueing}} + T_{\text{retransmission}} + T_{\text{os\_overhead}}$$
Our testbed tracks and records both metrics side-by-side to expose the real overhead of TCP framing and packet retransmission.

### 8. Concrete Project Data & Examples
- When tested over an unshaped direct Ethernet wire, an 80 KB compressed image transfers in $\approx 8 \text{ ms}$ (throughput $> 80 \text{ Mbps}$).
- When `tc/netem` limits bandwidth to $512 \text{ Kbps}$ with $50 \text{ ms}$ delay and $0.5\%$ loss:
  - $T_{\text{theoretical}} = \frac{80{,}000 \times 8}{512{,}000} = 1.25 \text{ seconds}$.
  - $T_{\text{measured}} \approx 1.38 \text{ seconds}$ (the additional $130 \text{ ms}$ reflects RTT delay, TCP handshakes, and netem queueing).

### 9. How to State This in the Viva
> *"To ensure rigorous scientific validity, we avoided in-memory simulated networks. Instead, we implemented a physical dual-laptop testbed. The sender runs on Ubuntu Linux because the Linux kernel provides native traffic control (`tc/netem`) at the network interface level to enforce deterministic bandwidth limits and packet drops. The receiver runs on Windows 11 equipped with an NVIDIA RTX 3050 GPU to simulate a hospital diagnostic server running PyTorch CUDA inference."*

### 10. Viva Questions & Model Answers
- **Q1: Why did you not run traffic shaping on Windows?**
  *Answer*: Windows does not provide kernel-level queuing disciplines equivalent to Linux `tc/netem`. Third-party Windows packet filters (like clumsy or WinDivert) operate in user space or through intermediate driver filters, which lack the reproducibility and fine-grained token-bucket rate limiting available natively in the Linux kernel.
- **Q2: Did you use a cross-over Ethernet cable or a switch?**
  *Answer*: Modern Gigabit Ethernet Network Interface Cards support Auto-MDI/MDIX, which automatically negotiates and detects cable transceivers. Therefore, a standard CAT5e or CAT6 Ethernet cable creates a direct peer-to-peer physical connection between both laptops without requiring a router or external switch.

---

# MODULE 3: DATASET ENGINEERING & PREPROCESSING PIPELINE

### 1. Fundamentals
Supervised medical image segmentation models require paired input images and pixel-accurate ground-truth binary masks. The standard benchmark in breast ultrasound AI research is the **Breast Ultrasound Images (BUSI)** dataset (published by Al-Dhabyani et al., 2020).

### 2. Why We Use This in This Project
The BUSI dataset provides a representative clinical distribution of ultrasound scans containing diverse lesion shapes, acoustic shadows, and tissue densities.

### 3. Exact Repository Implementation
- Dataset management: `sender/dataset_manager.py` (Lines 1-180)
- Splits definition: `data/BUSI/splits.json` (Lines 1-820)
- PyTorch dataset loader: `models/train_unet.py` (Lines 49-112)
- Multi-mask unification: `receiver/segmentation_metrics.py` (Lines 55-92)

### 4. Internal Mechanism & System Workflow
```
BUSI Dataset (780 Images)
├── Benign: 437 images
├── Malignant: 210 images
└── Normal: 133 images
            │
            ▼ Stratified 70 / 15 / 15 Partition
┌────────────────────────────────────────────────────────┐
│ - Training Set:   546 images (306 Benign, 147 Mal, 93 Normal) │
│ - Validation Set: 117 images ( 65 Benign,  32 Mal, 20 Normal) │
│ - Test Set:       117 images ( 66 Benign,  31 Mal, 20 Normal) │
└────────────────────────────────────────────────────────┘
```
1. **Directory Structure**: Scans are stored in `data/BUSI/benign/`, `data/BUSI/malignant/`, and `data/BUSI/normal/`.
2. **Multi-Lesion Mask Fusion**: Ultrasound scans often feature satellite lesions or multifocal tumor sites. In BUSI, these appear as separate mask files: `benign (10)_mask.png` and `benign (10)_mask_1.png`. If only the primary mask were loaded, downstream evaluation would penalize the model for detecting the second lesion. Our system scans for all matching mask stems and combines them using a pixel-wise logical maximum:
   $$M_{\text{combined}}(x, y) = \max_{k=1}^K M_k(x, y)$$
3. **Normal Images Policy**: Normal scans represent healthy breast tissue without focal lesions. There are no mask files on disk. The dataset manager assigns normal images a synthetic ground truth of all zeros ($256 \times 256$ zeros).
4. **Data Split Rigor**: The split was generated once using a stratified seed and locked into `splits.json`. The **117 held-out test images** were strictly never seen during training or validation checkpoint selection.

### 5. Input, Processing, and Output
- **Input**: Original ultrasound scan (PNG) and mask file(s).
- **Processing**: Filename stem parsing $\rightarrow$ Mask fusion $\rightarrow$ Resizing to $256 \times 256$ (bilinear for images, nearest-neighbor for masks to prevent boundary blurring) $\rightarrow$ Grayscale conversion $\rightarrow$ Normalization to $[0.0, 1.0]$.
- **Output**: Preprocessed image tensor $(1, 256, 256)$ and binary target tensor $(1, 256, 256)$.

### 6. Module Interconnections
- `sender/dataset_manager.py` exposes image paths and categories to the sender GUI.
- `models/train_unet.py` consumes the dataset via `splits.json` during training.
- `receiver/segmentation_metrics.py` locates ground-truth masks on the receiver node to evaluate incoming predictions.

### 7. Mathematical Foundation
Mask fusion is defined as the union of all $K$ annotated lesion regions:
$$M_{\text{fused}} = \bigcup_{k=1}^K M_k \iff M_{\text{fused}}(i, j) = \bigvee_{k=1}^K M_k(i, j) \in \{0, 1\}$$
Nearest-neighbor interpolation for mask resizing guarantees that:
$$M_{\text{resized}}(x, y) \in \{0, 1\} \quad \forall (x, y)$$
avoiding fractional boundary artifacts that occur with bilinear interpolation.

### 8. Concrete Project Data & Examples
- Sample `benign (10)` contains 2 mask files: `benign (10)_mask.png` and `benign (10)_mask_1.png`. In our pipeline, both masks are combined, creating an annotated ground truth with 2 distinct lesion regions.
- Out of 117 held-out test samples:
  - 66 are Benign
  - 31 are Malignant
  - 20 are Normal (zero-lesion controls)

### 9. How to State This in the Viva
> *"We utilized the benchmark BUSI dataset comprising 780 scans across benign, malignant, and normal categories. We enforced two critical preprocessing rules: First, for multi-lesion images, all masks are combined using pixel-wise logical OR to avoid penalizing the model for segmenting secondary lesions. Second, healthy normal scans are included with all-zero ground-truth masks to evaluate false-positive behavior. The dataset is strictly split into 70% train (546), 15% validation (117), and 15% held-out test (117) stored in `splits.json`."*

### 10. Viva Questions & Model Answers
- **Q1: Why did you resize images to $256 \times 256$?**
  *Answer*: Original ultrasound scans have irregular resolutions ranging from $500 \times 400$ to $1000 \times 800$. Deep learning convolutional architectures require consistent tensor dimensions for mini-batch tensor operations. $256 \times 256$ is the standard resolution in medical AI segmentation benchmarks that balances spatial resolution with GPU memory constraints and sub-10ms inference latency.
- **Q2: Why did you use nearest-neighbor interpolation for the masks instead of bilinear?**
  *Answer*: Masks are binary categorical labels ($0$ for background, $1$ for lesion). Bilinear or bicubic interpolation computes a weighted average of surrounding pixels, producing fractional floating-point values (e.g., $0.35, 0.72$) along edges, creating fuzzy, artificial boundaries. Nearest-neighbor preserves strict binary values ($0$ and $1$).

---

# MODULE 4: VARIABLE LOSSY JPEG COMPRESSION & IMAGE FIDELITY METRICS

### 1. Fundamentals
JPEG (Joint Photographic Experts Group) is the most widely adopted lossy image compression standard. It operates on $8 \times 8$ pixel blocks in the spatial frequency domain using the Discrete Cosine Transform (DCT).

### 2. Why We Use This in This Project
Ultrasound machines in remote areas can easily convert raw captures to standard JPEG without specialized hardware encoders. By tuning the JPEG quality factor $Q \in \{30, 50, 70, 90\}$, we can systematically evaluate the trade-off between image file size and downstream segmentation accuracy.

### 3. Exact Repository Implementation
- Compression engine: `sender/compression.py` (Lines 38-139)
- Distortion metrics (PSNR & SSIM): `sender/image_metrics.py` (Lines 1-55)
- Standard research qualities: `shared/constants.py` (`COMPRESSION_QUALITIES = [30, 50, 70, 90]`)

### 4. Internal Mechanism & System Workflow
1. **Discrete Cosine Transform (DCT)**: Converts each $8 \times 8$ spatial pixel block into 64 orthogonal frequency basis functions (1 DC coefficient and 63 AC coefficients).
2. **Quantization (The Lossy Step)**: Frequency coefficients are divided by values from a Quantization Table $Q_{\text{table}}$ scaled by the quality factor $Q$, and rounded to the nearest integer:
   $$F_Q(u, v) = \text{round}\left( \frac{F(u, v)}{Q_{\text{table}}(u, v) \cdot S(Q)} \right)$$
   Higher frequencies (fine speckle texture) are quantized more aggressively to zero.
3. **Entropy Encoding**: The quantized coefficients are zig-zag ordered and compressed using Run-Length Encoding (RLE) and Huffman coding.
4. **Strict Safety Policy**: Ground-truth masks are **never** subjected to lossy compression. The code explicitly raises a `ValueError` if a mask file is passed to `compress_image`.

### 5. Input, Processing, and Output
- **Input**: Original ultrasound scan (Path, NumPy array, or PIL Image) and quality factor $Q \in [1, 100]$.
- **Processing**: In-memory JPEG encoding via `io.BytesIO` $\rightarrow$ In-memory decode of compressed byte stream $\rightarrow$ Calculation of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
- **Output**: `CompressionResult` containing compressed bytes, compressed file size, compression ratio, decoded image array, PSNR (dB), and SSIM.

### 6. Module Interconnections
`sender/compression.py` provides compressed bytes to `sender/tcp_client.py` for wire transmission, and passes distortion metrics to `sender/experiment_orchestrator.py` for database logging.

### 7. Mathematical Foundation
#### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{M \cdot N} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I_{\text{orig}}(i, j) - I_{\text{comp}}(i, j)]^2$$

#### Peak Signal-to-Noise Ratio (PSNR)
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10} \left( \frac{255}{\sqrt{\text{MSE}}} \right) \quad (\text{for uint8}, \text{MAX}_I = 255)$$
- If $I_{\text{orig}} = I_{\text{comp}}$, $\text{MSE} = 0 \implies \text{PSNR} = \infty$.
- Higher PSNR indicates lower pixel distortion.

#### Structural Similarity Index Measure (SSIM)
Evaluates luminance ($l$), contrast ($c$), and structure ($s$):
$$\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
Where:
- $\mu_x, \mu_y$: Local mean pixel intensities.
- $\sigma_x^2, \sigma_y^2$: Local pixel variances.
- $\sigma_{xy}$: Cross-covariance between original and compressed patches.
- $C_1 = (k_1 L)^2, C_2 = (k_2 L)^2$ (stability constants preventing division by zero, where $L=255, k_1=0.01, k_2=0.03$).

### 8. Concrete Project Data & Examples
Measured compression metrics across the BUSI dataset:
| Quality | Typical Compressed Size | Compression Ratio | Average PSNR | Average SSIM | Visual Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original** | ~250 KB (PNG) | 1.0x (Baseline) | $\infty$ | 1.0000 | None (Raw acquisition) |
| **Q90** | ~65 KB | ~3.8x | ~41.5 dB | ~0.985 | Imperceptible distortion |
| **Q70** | ~38 KB | ~6.5x | ~36.8 dB | ~0.952 | Minor speckle smoothing |
| **Q50** | ~26 KB | ~9.6x | ~33.4 dB | ~0.915 | Noticeable block boundary artifacts |
| **Q30** | ~17 KB | ~14.7x | ~29.8 dB | ~0.842 | Heavy blocking, acoustic edge loss |

### 9. How to State This in the Viva
> *"We implemented variable-quality JPEG compression using 4 standard research quality tiers: Q90, Q70, Q50, and Q30. Compression is performed in memory to measure compression ratio, PSNR, and SSIM against the uncompressed image. PSNR measures pixel-level reconstruction error on a logarithmic decibel scale, while SSIM measures perceived structural correlation. Our results show that Q90 and Q70 retain over 95% structural similarity, whereas Q30 introduces high-frequency blocking artifacts that affect acoustic lesion boundaries."*

### 10. Viva Questions & Model Answers
- **Q1: Why is PSNR alone insufficient to evaluate medical image quality?**
  *Answer*: PSNR is based on Mean Squared Error (MSE), which computes pixel-by-pixel intensity differences independently. However, human visual perception and convolutional neural networks are sensitive to spatial structures, edges, and textures rather than absolute pixel intensity shifts. A slight global brightness shift causes a large drop in PSNR despite preserving all edges, whereas blur across a tumor boundary may yield a moderate PSNR drop while destroying diagnostic utility. SSIM models structural luminance and contrast correlation, making it a better metric for medical imaging.
- **Q2: Why did you choose JPEG instead of newer formats like WebP, HEIC, or JPEG 2000?**
  *Answer*: JPEG is universally supported across virtually all clinical hardware, legacy ultrasound consoles, and embedded PACS systems without requiring specialized proprietary codecs or heavy compute overhead. Testing JPEG provides a standardized, hardware-agnostic baseline that can be deployed on any mobile ultrasound unit worldwide.

---

# MODULE 5: LINUX KERNEL TRAFFIC CONTROL (`tc` / `netem`) & RESEARCH PROFILES

### 1. Fundamentals
The Linux kernel includes a Traffic Control (`tc`) subsystem within the `iproute2` package that regulates how packets are queued, prioritized, delayed, and dropped on network interface cards. The Network Emulator (`netem`) queuing discipline (`qdisc`) allows deterministic emulation of real-world wide area network (WAN) impairments directly inside the kernel.

### 2. Why We Use This in This Project
To test our adaptive compression algorithm under realistic telemedicine network conditions, we need repeatable network constraints: controlled bandwidth limits, propagation delays, and packet loss rates. Linux `tc/netem` provides this at the kernel level, ensuring packets experience real network queueing before being placed onto the physical Ethernet cable.

### 3. Exact Repository Implementation
- Network controller: `sender/network_control.py` (Lines 1-193)
- Profile definitions: `sender/network_control.py` (Lines 32-37)
- UI interface: `ui/sender/network_view.py` (Lines 1-220)

### 4. Internal Mechanism & System Workflow
```
User selects Network Profile in GUI
                 │
                 ▼
Subprocess executes: sudo tc qdisc del dev eth0 root (Clean previous rules)
                 │
                 ▼
Subprocess executes: sudo tc qdisc add dev eth0 root netem rate <BW> delay <D>ms loss <L>%
                 │
                 ▼
Kernel verifies active qdisc via: tc -s qdisc show dev eth0
                 │
                 ▼
Status returned: Profile active, kernel verified
```
1. Before applying new network rules, the controller issues `tc qdisc del dev <interface> root` to purge existing rules and avoid conflicting nested queues.
2. The controller adds a root `netem` qdisc specifying:
   - `rate`: Bandwidth limitation (e.g., `10mbit`, `2mbit`, `512kbit`, `128kbit`).
   - `delay`: Packet transmission latency in milliseconds (e.g., `5ms`, `20ms`, `50ms`, `100ms`).
   - `loss`: Random packet drop percentage (e.g., `0.0%`, `0.5%`, `1.0%`, `2.0%`).
3. The controller verifies enforcement by parsing the output of `tc -s qdisc show dev <interface>` via regular expressions.

### 5. Input, Processing, and Output
- **Input**: Interface name (e.g., `eth0`), bandwidth string, delay ($ms$), and loss percentage ($\%$).
- **Processing**: Execution of Linux system commands with root privileges (`sudo tc`) $\rightarrow$ Parsing kernel queue statistics.
- **Output**: Active profile state confirmed, theoretical transmission delay computed.

### 6. Module Interconnections
`sender/network_control.py` shapes the physical network interface before `sender/tcp_client.py` transmits image bytes, directly influencing measured throughput and triggering quality adjustments in `sender/adaptive_controller.py`.

### 7. Mathematical Foundation
The theoretical transmission time across a shaped link is:
$$T_{\text{theoretical}} = \frac{\text{Payload Bytes} \times 8}{\text{Bandwidth (bps)}}$$
When propagation delay $D$ and packet loss rate $p$ are introduced, the expected completion time over TCP expands due to packet retransmissions and RTT waits:
$$E[T_{\text{transmission}}] \approx T_{\text{theoretical}} + 2D + \left( \frac{p}{1 - p} \right) \cdot \text{RTO}$$
Where $\text{RTO}$ is the TCP Retransmission Time-Out.

### 8. Concrete Project Data & Research Profiles
We standardized 5 research network profiles representing real clinical telemedicine deployment environments:
| Profile Identifier | Bandwidth | Added Delay | Packet Loss | Real-World Telemedicine Analogy |
| :--- | :--- | :--- | :--- | :--- |
| **High Speed** | 10 Mbps | 5 ms | 0.0% | Hospital Campus LAN / Urban Dedicated Fiber |
| **Medium Speed** | 2 Mbps | 20 ms | 0.5% | Standard Suburban 4G LTE / Regional Telehealth Link |
| **Constrained** | 512 Kbps | 50 ms | 1.0% | Rural 3G Mobile Health Van / Congested Microwave Link |
| **Severely Constrained** | 128 Kbps | 100 ms | 2.0% | Emergency Field Satellite (VSAT) / Disaster Zone Link |
| **Unshaped** | Wire Speed | 0 ms | 0.0% | Direct 1 Gbps Ethernet Cable (Baseline Control) |

### 9. How to State This in the Viva
> *"To rigorously evaluate transmission performance, we implemented network emulation using the Linux kernel's Traffic Control (`tc`) and Network Emulator (`netem`) subsystem. This controls bandwidth, delay, and packet loss at the kernel queue level on the physical interface. We established four standardized telemedicine profiles ranging from 10 Mbps campus fiber down to 128 Kbps emergency satellite links, allowing us to evaluate system behavior under real-world connectivity constraints."*

### 10. Viva Questions & Model Answers
- **Q1: Why does `tc/netem` require `sudo` permissions?**
  *Answer*: Network queueing disciplines operate in Linux kernel space. Altering packet scheduling policies or dropping packets at the Network Interface Card directly impacts host network stack integrity, which is restricted to administrative (`root`) privileges.
- **Q2: How does TCP handle the packet loss configured in `netem`?**
  *Answer*: When `netem` drops a packet, the sender fails to receive an acknowledgment. After either receiving 3 duplicate ACKs (Fast Retransmit) or suffering a Retransmission Timeout (RTO), TCP retransmits the missing segment and reduces its congestion window (`cwnd`), causing a drop in instantaneous throughput. This drop is captured by our adaptive controller's EWMA tracker.

---

# MODULE 6: CUSTOM TCP WIRE PROTOCOL & CRYPTOGRAPHIC INTEGRITY

### 1. Fundamentals
Application-layer network protocols define how structured messages are serialized into byte streams, transmitted over transport protocols (TCP), and parsed at the receiver. Because TCP is a stream-oriented protocol without inherent message boundaries, the application layer must implement **message framing** to prevent data corruption from packet fragmentation or coalescing.

### 2. Why We Use This in This Project
High-level application protocols like HTTP/REST or gRPC introduce protocol overhead: chunked transfer headers, HTTP verb parsing, and cookie/metadata overhead. In low-bandwidth channels (e.g., 128 Kbps), every byte of overhead increases transmission latency.

We designed a lightweight binary framing protocol (`NAC1` for transmission, `NACK` for acknowledgment) with:
1. Fixed-size binary headers for $O(1)$ parsing.
2. Complete payload reception handling (`recv_exact`) to address TCP stream fragmentation.
3. End-to-end cryptographic SHA-256 integrity verification.
4. Denial-of-Service bounds preventing memory allocation attacks.

### 3. Exact Repository Implementation
- Protocol definition and socket primitives: `shared/protocol.py` (Lines 1-231)
- Metadata structures: `shared/metadata.py` (Lines 1-85)
- Protocol constants: `shared/constants.py` (Lines 1-50)
- Server-side frame processing: `receiver/tcp_server.py` (Lines 110-185)

### 4. Internal Mechanism & Wire Framing Architecture
#### Image Transmission Frame (`NAC1`)
Total Header Size = 14 bytes (`IMAGE_HEADER_FORMAT = "!4sHII"`)
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 MAGIC_BYTES: b"NAC1" (4 bytes)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Protocol Version (uint16)    |  Metadata JSON Length N (u32) |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Metadata JSON Length (cont)     |   Image Payload Len M |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Image Payload Length M (cont)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               N bytes: UTF-8 Metadata JSON String             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               M bytes: Raw Compressed Image Binary Payload    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Acknowledgment Frame (`NACK`)
Total Header Size = 10 bytes (`ACK_HEADER_FORMAT = "!4sHI"`)
```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                ACK_MAGIC: b"NACK" (4 bytes)                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Protocol Version (uint16)    |  ACK JSON Length K (uint32)   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               K bytes: UTF-8 Acknowledgment JSON String       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Handling TCP Stream Fragmentation (`recv_exact`)
A common socket programming bug is assuming `sock.recv(N)` returns $N$ bytes. Under network congestion, `recv()` often returns partial chunks (e.g., 1440 bytes). Our implementation wraps reads in `recv_exact`:
```python
def recv_exact(sock: socket.socket, num_bytes: int, timeout: Optional[float] = None) -> bytes:
    buffer = bytearray()
    while len(buffer) < num_bytes:
        chunk = sock.recv(min(CHUNK_SIZE, num_bytes - len(buffer)))
        if not chunk:
            raise ConnectionError("Socket closed prematurely by peer.")
        buffer.extend(chunk)
    return bytes(buffer)
```
This guarantees complete frame assembly regardless of network fragmentation.

### 5. Input, Processing, and Output
- **Input**: Transmission metadata (image ID, experiment ID, quality, SHA-256 hash) and compressed image bytes.
- **Processing**: Packing struct header $\rightarrow$ Sending header, metadata JSON, and chunked image binary $\rightarrow$ Receiving ACK frame and parsing verification status.
- **Output**: Verified transmission confirmation, measured wire duration, and throughput.

### 6. Module Interconnections
- `sender/tcp_client.py` uses `send_image_frame()` and `recv_ack_frame()`.
- `receiver/tcp_server.py` uses `recv_image_frame()` and `send_ack_frame()`.

### 7. Mathematical Foundation & Cryptographic Verification
To guarantee that not a single bit of image data was corrupted across the network:
$$\text{SHA-256}(B) = \text{SHA-256-Block-Transform}(B) \in \{0, 1\}^{256}$$
Sender calculates the hash over compressed bytes $B_{\text{sent}}$:
$$H_{\text{sender}} = \text{hex}(\text{SHA-256}(B_{\text{sent}}))$$
Receiver calculates the hash over received bytes $B_{\text{recv}}$:
$$H_{\text{receiver}} = \text{hex}(\text{SHA-256}(B_{\text{recv}}))$$
Integrity validation condition:
$$H_{\text{sender}} == H_{\text{receiver}}$$
If the digests differ by even 1 bit, the receiver rejects the payload, emits `ACK_CHECKSUM_MISMATCH`, and refuses to run inference on corrupted data.

### 8. Concrete Project Data & Examples
- **Header efficiency**: The 14-byte `NAC1` header accounts for only $0.02\%$ of an average 65 KB image payload.
- **Security bounds**:
  - `MAX_METADATA_SIZE = 64 * 1024` (64 KB maximum metadata).
  - `MAX_IMAGE_PAYLOAD_SIZE = 50 * 1024 * 1024` (50 MB maximum payload).
  - If a corrupted packet specifies an image length of $2 \text{ GB}$, the receiver immediately aborts with `ValueError` rather than attempting a massive memory allocation.

### 9. How to State This in the Viva
> *"We designed a custom binary TCP framing protocol named `NAC1` with a 14-byte network-byte-order header containing magic bytes, protocol version, metadata length, and payload length. To handle TCP stream fragmentation, we implemented a robust `recv_exact` loop that reads exact byte boundaries. Payload integrity is validated using end-to-end SHA-256 cryptographic hashes: the sender transmits the hash in metadata, and the receiver verifies the received bytes before saving or running model inference."*

### 10. Viva Questions & Model Answers
- **Q1: Why not use HTTP or WebSockets instead of a custom binary protocol?**
  *Answer*: HTTP and WebSockets introduce significant parsing and transmission overhead: string-based HTTP headers, HTTP chunking, frame masking, and cookie handling. In telemedicine over 128 Kbps links, this overhead increases latency. Our binary protocol has only 14 bytes of framing overhead and allows deterministic microsecond-level socket benchmarking.
- **Q2: What happens if the network cable is unplugged mid-transmission?**
  *Answer*: In our `recv_exact` function, if the socket is closed or disconnected, `sock.recv()` returns an empty byte string `b""` or raises a `socket.timeout`. The system immediately catches this exception, cleans up socket descriptors, logs a failed transmission, and informs the user through the GUI without crashing.

---

# MODULE 7: BANDWIDTH-ADAPTIVE RATE CONTROLLER

### 1. Fundamentals
Rate adaptation algorithms dynamically adjust data transmission rates or compression quality to match the available bandwidth of a communication channel. The goal is to maximize throughput and fidelity while avoiding network congestion, packet queuing, and packet loss.

### 2. Why We Use This in This Project
In tele-ultrasound transmission across mobile networks, bandwidth fluctuates as the ambulance moves or network traffic varies. If the sender continuously transmits high-quality images (Q90) across a degraded link, the transmission queue fills, latency spikes, and socket timeouts occur. Conversely, transmitting low-quality images (Q30) across high-speed connections unnecessarily degrades diagnostic accuracy.

Our adaptive controller dynamically selects the optimal JPEG quality based on real-time channel throughput.

### 3. Exact Repository Implementation
- Rate controller implementation: `sender/adaptive_controller.py` (Lines 1-120)
- Controller thresholds: `sender/adaptive_controller.py` (Lines 26-30)
- Decision history and EWMA update: `sender/adaptive_controller.py` (Lines 47-110)
- UI visualization: `ui/sender/adaptive_view.py` (Lines 1-220)

### 4. Internal Mechanism & System Workflow
```
Transmission Completes (Bytes M, Measured Duration T seconds)
                         │
                         ▼
             Calculate Instantaneous Rate:
                 R_t = (M * 8) / (T * 1000)  [Kbps]
                         │
                         ▼
        Update Exponential Moving Average (EWMA):
       EWMA_t = alpha * R_t + (1 - alpha) * EWMA_{t-1}
                         │
                         ▼
    Evaluate against Bandwidth Thresholds:
    ├── EWMA >= 10,000 Kbps (10 Mbps) ────► Select Q90
    ├── 2,000 <= EWMA < 10,000 Kbps ──────► Select Q70
    ├── 512 <= EWMA < 2,000 Kbps ─────────► Select Q50
    └── EWMA < 512 Kbps ──────────────────► Select Q30
                         │
                         ▼
   Return AdaptiveDecision(quality, bandwidth, explanation)
```
1. **Passive Throughput Estimation**: Rather than sending synthetic "probe packets" that consume channel bandwidth, our controller measures the actual throughput of previous image transmissions.
2. **EWMA Smoothing**: To prevent rapid oscillations caused by momentary network spikes, measurements are smoothed using an Exponentially Weighted Moving Average (EWMA) with smoothing coefficient $\alpha = 0.3$.
3. **Threshold Decision**: The smoothed bandwidth estimate is mapped to our quality tiers.

### 5. Input, Processing, and Output
- **Input**: Measured payload byte size $M$ and measured physical transmission duration $T_{\text{measured}}$ (seconds).
- **Processing**: Instantaneous throughput calculation $\rightarrow$ EWMA smoothing update $\rightarrow$ Threshold comparison.
- **Output**: `AdaptiveDecision` object containing selected quality factor, estimated bandwidth, operation mode, and explanation text.

### 6. Module Interconnections
`sender/experiment_orchestrator.py` captures transmission times from `sender/tcp_client.py`, feeds them into `sender/adaptive_controller.py`, and applies the resulting quality factor to the next image compressed by `sender/compression.py`.

### 7. Mathematical Foundation
Instantaneous throughput:
$$R_t = \frac{M_{\text{bytes}} \times 8}{T_{\text{sec}} \times 1000} \quad [\text{Kbps}]$$

Exponentially Weighted Moving Average (EWMA):
$$\text{EWMA}_t = \alpha \cdot R_t + (1 - \alpha) \cdot \text{EWMA}_{t-1}$$
Where $\alpha \in (0, 1]$ is the smoothing weight (set to $\alpha = 0.3$).
- A lower $\alpha$ (e.g., 0.1) creates high smoothing but reacts slowly to bandwidth drops.
- A higher $\alpha$ (e.g., 0.8) reacts quickly to drops but causes quality oscillations from transient network jitter.
- $\alpha = 0.3$ provides an appropriate balance for medical image transmission.

### 8. Concrete Project Data & Decision Thresholds
| Bandwidth Range | Quality Tier | Clinical Rationale | Expected Latency |
| :--- | :--- | :--- | :--- |
| $\ge 10\text{ Mbps}$ | **Q90** | High bandwidth: prioritize diagnostic fidelity; transmission latency is sub-100ms. | $< 0.10 \text{ s}$ |
| $2\text{ Mbps} \le B < 10\text{ Mbps}$ | **Q70** | Moderate broadband: balanced compression; preserves 95% SSIM while halving file size. | $0.15 - 0.30 \text{ s}$ |
| $512\text{ Kbps} \le B < 2\text{ Mbps}$ | **Q50** | Constrained link: reduce file size by ~90% to prevent socket buffer bloat. | $0.30 - 0.60 \text{ s}$ |
| $< 512\text{ Kbps}$ | **Q30** | Severely constrained: minimize payload to maintain real-time responsiveness. | $0.50 - 1.20 \text{ s}$ |

### 9. How to State This in the Viva
> *"Our bandwidth-adaptive rate controller uses passive throughput monitoring to avoid wasting bandwidth on synthetic probe packets. It calculates instantaneous throughput from each transmission and smooths the estimate using an Exponentially Weighted Moving Average with alpha = 0.3. This estimate is mapped to our quality tiers: Q90 for links of 10 Mbps or higher, Q70 down to 2 Mbps, Q50 down to 512 Kbps, and Q30 below 512 Kbps. This keeps transmission latency low while maximizing image fidelity."*

### 10. Viva Questions & Model Answers
- **Q1: Why did you use passive throughput estimation instead of active probing like `iperf`?**
  *Answer*: Active network probing generates dummy traffic that consumes valuable channel capacity. In a severely constrained 128 Kbps or 512 Kbps link, running active probe packets would cause congestion and delay the actual medical image transmission. Passive monitoring measures only genuine application data transfers without adding network overhead.
- **Q2: How does the system avoid quality oscillation if throughput fluctuates around a threshold?**
  *Answer*: We mitigate oscillation in two ways: first, the EWMA smoothing factor ($\alpha = 0.3$) filters out single-packet throughput spikes. Second, the discrete quality tiers provide natural separation buffers between the 512 Kbps, 2 Mbps, and 10 Mbps operating points.

---

# MODULE 8: DEEP LEARNING LESION SEGMENTATION ENGINE (U-NET)

### 1. Fundamentals
Medical image segmentation is the task of classifying every pixel in an ultrasound scan as either lesion tissue or background tissue. The **U-Net** architecture (Ronneberger et al., MICCAI 2015) is the benchmark standard for medical image segmentation due to its symmetric encoder-decoder structure and skip connections.

### 2. Why We Use This in This Project
U-Net's skip connections transfer high-resolution spatial feature maps directly from the encoder stages to the corresponding decoder stages. This allows the network to retain fine boundary details that are otherwise lost during pooling operations.

To evaluate the true impact of image compression, we use a **strictly frozen U-Net model** on the receiver node.

### 3. Exact Repository Implementation
- Model architecture: `models/unet.py` (Lines 1-106)
- Checkpoint manager: `receiver/model_manager.py` (Lines 1-152)
- Inference engine: `receiver/segmentation.py` (Lines 1-176)
- Training pipeline: `models/train_unet.py` (Lines 1-409)
- Checkpoint file: `models/best_unet.pth` (Epoch 21 checkpoint)

### 4. Internal Mechanism & Layer-by-Layer Architecture
```
INPUT: (B, 1, 256, 256) Normalized Float32 Tensor [0.0, 1.0]
 │
 ├── [inc]   DoubleConv(1 -> 32)       ──► (B, 32, 256, 256) ───[Skip 1]───┐
 ├── [down1] MaxPool + DoubleConv      ──► (B, 64, 128, 128) ───[Skip 2]─┐ │
 ├── [down2] MaxPool + DoubleConv      ──► (B, 128, 64, 64)  ───[Skip 3]┐│ │
 ├── [down3] MaxPool + DoubleConv      ──► (B, 256, 32, 32)  ───[Skip 4]││ │
 ├── [down4] MaxPool + DoubleConv      ──► (B, 512, 16, 16) [Bottleneck]││ │
 │                                                                       ││ │
 ├── [up1]   ConvTranspose + Skip4 Cat ──► (B, 256, 32, 32) ◄────────────┘│ │
 ├── [up2]   ConvTranspose + Skip3 Cat ──► (B, 128, 64, 64) ◄─────────────┘ │
 ├── [up3]   ConvTranspose + Skip2 Cat ──► (B, 64, 128, 128) ◄──────────────┘
 ├── [up4]   ConvTranspose + Skip1 Cat ──► (B, 32, 256, 256) ◄───────────────┘
 └── [outc]  Conv2d(32 -> 1, 1x1 kernel)──► (B, 1, 256, 256) Raw Logits
                                                 │
                                                 ▼
               Sigmoid Activation: p = 1 / (1 + exp(-logits))
                                                 │
                                                 ▼
             Binary Thresholding at tau = 0.5: Mask = (p >= 0.5)
```

#### Frozen Model Policy
To ensure scientific validity, the receiver model is strictly frozen:
```python
self.model.eval()
for p in self.model.parameters():
    p.requires_grad = False
```
With weights frozen, any change in segmentation output across experiments is strictly caused by compression artifacts, not model training variance.

### 5. Input, Processing, and Output
- **Input**: Received ultrasound image bytes.
- **Processing**: Decoding to BGR array $\rightarrow$ Grayscale conversion $\rightarrow$ Resize to $256 \times 256$ $\rightarrow$ Normalize to $[0.0, 1.0]$ $\rightarrow$ Tensor conversion $(1, 1, 256, 256)$ $\rightarrow$ PyTorch CUDA inference under `torch.no_grad()` $\rightarrow$ Sigmoid activation $\rightarrow$ Thresholding at $\tau = 0.5$.
- **Output**: Binary lesion mask $(256 \times 256)$, continuous probability map, RGB lesion contour overlay, and measured inference duration ($ms$).

### 6. Module Interconnections
`receiver/tcp_server.py` delivers saved image bytes to `receiver/segmentation.py`, which executes the model maintained by `receiver/model_manager.py` and passes the resulting mask to `receiver/segmentation_metrics.py`.

### 7. Mathematical Foundation
#### Combined Loss Function (Used in Training)
The model was trained using a combination of Binary Cross-Entropy (BCE) with Logits and Soft Dice Loss:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
Where:
$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$
$$\mathcal{L}_{\text{Dice}}(y, \hat{y}) = 1 - \frac{2 \sum_{i=1}^N \sigma(z_i) y_i + \epsilon}{\sum_{i=1}^N \sigma(z_i) + \sum_{i=1}^N y_i + \epsilon}$$
Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation, and $\epsilon = 1.0$ is the Laplace smoothing parameter preventing division by zero.

### 8. Concrete Project Data & Verified Training History
- **Device Used**: NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- **Trainable Parameters**: **7,762,465** (~7.76 million parameters).
- **Training Duration**: 25 epochs completed in **453.71 seconds** (~7.56 minutes).
- **Optimizer & Scheduler**: AdamW ($\text{initial lr} = 0.001$, weight decay = $10^{-4}$) with Cosine Annealing learning rate schedule ($\eta_{\text{min}} = 10^{-5}$).
- **Best Validation Checkpoint**: **Epoch 21**
  - **Validation Dice**: **0.6056**
  - **Validation IoU**: **0.5287**
  - **Validation Loss**: **0.3830**
  - Saved to: `models/best_unet.pth`

### 9. How to State This in the Viva
> *"Our lesion segmentation engine implements a standard PyTorch U-Net with 4 downsampling encoder blocks, a 512-channel bottleneck, 4 upsampling decoder blocks with skip connections, and a 1x1 convolution output head. It has 7,762,465 trainable parameters. It was trained using a compound BCE and Soft Dice loss with AdamW and Cosine Annealing. The best checkpoint was achieved at Epoch 21 with a validation Dice of 0.6056. On the receiver, the model is strictly frozen using `model.eval()` and `requires_grad=False`, ensuring that any changes in segmentation performance across experiments are solely due to compression degradation."*

### 10. Viva Questions & Model Answers
- **Q1: Why did you combine BCE and Dice Loss during training?**
  *Answer*: Binary Cross-Entropy evaluates classification error on a per-pixel basis, which provides smooth gradients during early training epochs. However, ultrasound images suffer from severe class imbalance: lesions often occupy less than 10% of the image area, with healthy tissue filling the remaining 90%. BCE alone can bias the model toward predicting background everywhere. Dice Loss directly optimizes region overlap, addressing the class imbalance issue. Combining them yields both stable gradient flow and balanced region segmentation.
- **Q2: Why is the model run with `torch.no_grad()` on the receiver?**
  *Answer*: During inference, we do not perform backpropagation or weight updates. Using `torch.no_grad()` disables PyTorch's computational autograd graph construction, reducing GPU memory consumption by more than 50% and accelerating inference to under 10 milliseconds per image.
- **Q3: Can we optimize the loss weighting parameter $\lambda$ dynamically using Multi-Gradient Descent (MGDA)?**
  *Answer*: Yes. While a naive scalar minimization $\min_\lambda [\lambda L_{\text{Dice}} + (1-\lambda)L_{\text{BCE}}]$ collapses to a boundary extreme $\min(L_{\text{Dice}}, L_{\text{BCE}})$, the Multiple Gradient Descent Algorithm (MGDA, Sener & Koltun 2018) solves the minimum-norm problem directly in the gradient parameter space:
  $$\min_{0 \le \lambda \le 1} \|\lambda \nabla L_{\text{Dice}} + (1 - \lambda) \nabla L_{\text{BCE}}\|^2$$
  The closed-form analytical solution is $\hat{\lambda} = \frac{\|g_2\|^2 - g_1^T g_2}{\|g_1 - g_2\|^2}$, clipped to $[0, 1]$. At every mini-batch, this dynamically projects the descent direction onto the Pareto front, ensuring that both Dice and BCE losses decrease simultaneously ($\langle g_1, g^* \rangle = \langle g_2, g^* \rangle > 0$) without gradient cancellation. Our training suite supports this mode via `python models/train_unet.py --mgda`.

---

# MODULE 9: CLINICAL SEGMENTATION EVALUATION & EMPTY MASK POLICY

### 1. Fundamentals
Evaluating medical image segmentation requires overlap metrics that quantify how closely predicted boundaries match expert radiologist ground truth. The two standard metrics are the **Dice Similarity Coefficient (DSC)** and **Intersection over Union (IoU)**.

### 2. Why We Use This in This Project
These metrics allow us to quantitatively determine how much each level of lossy compression affects segmentation accuracy compared to the uncompressed baseline.

### 3. Exact Repository Implementation
- Metric computation: `receiver/segmentation_metrics.py` (Lines 18-54)
- Comparative color overlay: `receiver/segmentation_metrics.py` (Lines 94-134)
- Baseline evaluation suite: `experiments/evaluate_baseline.py` (Lines 1-210)
- Verified baseline results: `results/test_baseline_summary.json` (Lines 1-62)

### 4. Internal Mechanism & The Empty Mask Policy
A critical issue in medical segmentation research is handling **Normal (healthy) scans** that contain no lesions ($|G| = 0$). In standard Dice and IoU formulas, an empty ground truth produces $\frac{0}{0}$ (undefined division).

Our system implements a standardized **Clinical Empty Mask Policy**:
```
Case 1: Ground Truth is Empty (|G| = 0) and Prediction is Empty (|P| = 0)
        ──► Perfect True Negative (TN): Dice = 1.0, IoU = 1.0

Case 2: Ground Truth is Empty (|G| = 0) but Prediction has Lesion (|P| > 0)
        ──► False Positive Alarm (FP): Dice = 0.0, IoU = 0.0

Case 3: Ground Truth has Lesion (|G| > 0) but Prediction is Empty (|P| = 0)
        ──► False Negative Miss (FN): Dice = 0.0, IoU = 0.0

Case 4: Ground Truth has Lesion (|G| > 0) and Prediction has Lesion (|P| > 0)
        ──► Standard Overlap: Calculate exact Dice and IoU
```

### 5. Input, Processing, and Output
- **Input**: Predicted binary mask $P \in \{0, 1\}^{256 \times 256}$ and ground-truth binary mask $G \in \{0, 1\}^{256 \times 256}$.
- **Processing**: Summing True Positives ($\text{TP} = P \cap G$), False Positives ($\text{FP} = P \setminus G$), and False Negatives ($\text{FN} = G \setminus P$) $\rightarrow$ Applying overlap formulas $\rightarrow$ Generating a three-color comparative overlay.
- **Output**: Numeric Dice score $[0.0, 1.0]$, IoU score $[0.0, 1.0]$, and an RGB visualization.

### 6. Module Interconnections
`receiver/segmentation.py` produces the predicted mask, `receiver/segmentation_metrics.py` calculates metrics against ground-truth files located on disk, and `common/results_manager.py` writes the scores to the SQLite and CSV logs.

### 7. Mathematical Foundation
#### Dice Similarity Coefficient (DSC / F1-Score)
$$\text{Dice}(P, G) = \frac{2 \cdot |P \cap G|}{|P| + |G|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

#### Intersection over Union (IoU / Jaccard Index)
$$\text{IoU}(P, G) = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

#### Direct Mathematical Relationship Between Dice and IoU
$$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}} \iff \text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$
Because $\text{Dice} \ge \text{IoU}$ holds for all valid non-empty inputs, Dice is consistently higher than or equal to IoU.

#### Visual Comparative Color Demarcation
The overlay maps clinical prediction accuracy across three colors:
- **Yellow (Overlap)**: True Positives ($\text{TP} = P \cap G$).
- **Red (Over-segmentation)**: False Positives ($\text{FP} = P \cap \neg G$).
- **Green (Under-segmentation)**: False Negatives ($\text{FN} = \neg P \cap G$).

### 8. Concrete Project Data: Verified Baseline Test Results
The uncompressed baseline evaluation was run on all **117 held-out test scans** using the frozen checkpoint `best_unet.pth` on the RTX 3050 GPU (recorded in `results/test_baseline_summary.json`):
- **Total Test Images Evaluated**: 117
- **Total Skipped Images**: 0
- **Total Evaluation Duration**: 3.50 seconds
- **Average Inference Latency**: **6.95 ms per image**

| Evaluation Cohort | Sample Count | Mean Dice ($\pm$ Std) | Median Dice | Mean IoU ($\pm$ Std) | Median IoU | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All Test Samples** | **117** | **0.6552** $\pm 0.348$ | **0.8091** | **0.5733** $\pm 0.341$ | **0.6793** | **6.95 ms** |
| **Lesion Cases Only** | **97** | **0.6254** $\pm 0.328$ | **0.7787** | **0.5266** $\pm 0.308$ | **0.6376** | **7.64 ms** |
| **- Benign Cohort** | 66 | **0.6414** $\pm 0.330$ | **0.8015** | **0.5462** $\pm 0.314$ | **0.6688** | 9.97 ms |
| **- Malignant Cohort**| 31 | **0.5913** $\pm 0.325$ | **0.7592** | **0.4848** $\pm 0.295$ | **0.6119** | 2.70 ms |
| **- Normal Cohort** | 20 | **0.8000** $\pm 0.410$ | **1.0000** | **0.8000** $\pm 0.410$ | **1.0000** | 3.60 ms |

#### Key Analytical Insights
1. **Median vs Mean**: The overall median Dice is **0.8091**, which is significantly higher than the mean of **0.6552**. This indicates that the model segments the majority of ultrasound lesions with high accuracy ($> 80\%$), while a small number of challenging edge-case scans pull down the arithmetic mean.
2. **Normal Tissue Specificity**: In 16 out of 20 normal test scans (80%), the model correctly predicted no lesion, achieving a perfect Dice score of 1.0. In 4 scans, false-positive regions were detected, yielding a score of 0.0, resulting in the $80\%$ mean score for this cohort.

### 9. How to State This in the Viva
> *"We evaluate lesion segmentation using Dice Similarity Coefficient and Intersection over Union. A key contribution of our evaluation pipeline is our explicit Empty Mask Policy for healthy normal scans: when both ground truth and prediction are empty, it counts as a perfect True Negative (Dice=1.0); if false positive pixels are predicted, it receives Dice=0.0. On our 117 held-out test scans, the frozen baseline model achieves a mean Dice of 0.6552, a median Dice of 0.8091, and an average inference time of 6.95 milliseconds on the RTX 3050 GPU."*

### 10. Viva Questions & Model Answers
- **Q1: Why is median Dice higher than mean Dice in your results?**
  *Answer*: In medical image segmentation, Dice distributions are often bimodal. On the vast majority of clear benign and malignant scans, the model segments boundaries accurately, yielding Dice scores between 0.75 and 0.92. However, a minority of difficult cases—such as ill-defined malignant margins or small false-positive regions on normal scans—receive low scores (0.0 to 0.2). These outliers pull down the arithmetic mean, while the median (0.8091) reflects typical real-world performance.
- **Q2: Why is Dice always mathematically greater than or equal to IoU?**
  *Answer*: From the algebraic relationship $\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$, since $\text{Dice} \le 1.0$, the denominator $(2 - \text{Dice})$ is always $\ge 1.0$. Dividing Dice by a value greater than or equal to 1 produces a value less than or equal to Dice. They are equal only when both are 0 or both are 1.

---

# MODULE 10: END-TO-END AUTOMATIC WORKFLOW & BATCH EXECUTION

### 1. Fundamentals
Automated experiment orchestration coordinates multiple system operations—dataset iteration, image compression, network transmission, socket reception, integrity validation, model inference, metric computation, and database persistence—without manual intervention between images.

### 2. Why We Use This in This Project
Manually testing hundreds of images across multiple compression qualities and network profiles would be slow, error-prone, and difficult to reproduce. Our automated batch runner allows researchers to execute standardized multi-sample sweeps across defined experimental configurations with full traceability.

### 3. Exact Repository Implementation
- Experiment orchestrator: `sender/experiment_orchestrator.py` (Lines 1-320)
- Results view & batch trigger: `ui/sender/results_view.py` (Lines 1-210)
- Database persistence: `common/results_manager.py` (Lines 1-282)

### 4. Step-by-Step Execution Sequence
```
Step 1: Sender GUI initiates batch experiment (e.g. 10 test samples)
Step 2: Orchestrator sets up network profile via Linux tc/netem
Step 3: For each sample:
        a. Load image from BUSI test split
        b. Select quality factor (fixed tier or adaptive controller)
        c. Compress image in memory, compute PSNR and SSIM
        d. Compute SHA-256 hash of compressed payload
        e. Pack into NAC1 frame and transmit over TCP socket
Step 4: Receiver socket daemon ingests frame via recv_exact
Step 5: Receiver re-computes SHA-256 and verifies integrity match
Step 6: Receiver writes verified file to data/received/
Step 7: Frozen U-Net runs CUDA inference (sub-10ms)
Step 8: Receiver locates ground-truth mask, computes Dice and IoU
Step 9: Receiver sends NACK frame back with ACK_OK confirmation
Step 10: Both nodes record experiment row in SQLite and CSV
Step 11: GUI dashboards update live cards, progress bars, and tables
```

### 5. Input, Processing, and Output
- **Input**: Image selection mode (Single, First 10 Test Cases, Full Test Split), Quality Strategy (Fixed Quality vs Adaptive Mode), and Network Profile.
- **Processing**: Sequential pipeline execution with per-sample error handling, thread-safe cancellation checking (`is_cancelled` flag), and socket lifecycle management.
- **Output**: Synchronized database records, persisted image files, and updated GUI views.

### 6. Module Interconnections
The orchestrator acts as the central coordinator on the sender node, interfacing with `dataset_manager`, `compression`, `image_metrics`, `network_control`, `adaptive_controller`, and `tcp_client`.

### 7. Mathematical Foundation
Each experiment is assigned an atomic, collision-free identifier:
$$\text{Experiment ID} = \text{Prefix} + \text{"\_"} + \text{Timestamp}_{\text{ms}} + \text{"\_Q"} + \text{Quality}$$
Example: `EXP_20261112_235812_341_Q70`

This format guarantees temporal ordering and prevents database primary-key collisions even during rapid batch transmissions.

### 8. Concrete Project Data & Example Execution
- The **"First 10 Test Cases"** batch runner option selects the first 10 images from `splits.json['test']`.
- Under the "Typical Broadband (2 Mbps)" profile with Adaptive Mode, the batch completes in $\approx 6.5$ seconds:
  - 10 images compressed to Q70.
  - 10 SHA-256 hashes generated and verified.
  - 10 U-Net inferences executed in $\approx 70 \text{ ms}$ total GPU time.
  - 10 records committed to SQLite and CSV without manual intervention.

### 9. How to State This in the Viva
> *"We implemented an automated experiment orchestrator that enables reproducible batch evaluations. When launched, the orchestrator configures the network profile, loops through the selected test dataset, compresses each image, transmits the payload with SHA-256 verification, triggers GPU inference on the receiver, computes segmentation metrics, and logs the results to SQLite and CSV. It also includes thread-safe cancellation and error handling to ensure clean socket shutdown if an experiment is stopped."*

### 10. Viva Questions & Model Answers
- **Q1: What happens if an image transfer fails midway through a 100-image batch?**
  *Answer*: The orchestrator wraps each image transmission in a dedicated `try-except` block. If an individual transfer times out or suffers a socket error, the failure is logged with an error note in the database, resources are cleaned up, and the batch runner continues to the next image rather than crashing the application.
- **Q2: How do you prevent database locking when both sender and receiver log results?**
  *Answer*: Each node maintains its own local SQLite database file and CSV log. The sender logs transmission and compression metrics, while the receiver logs reception, checksum verification, and segmentation metrics. This physical separation prevents concurrent write locks.

---

# MODULE 11: SYSTEM HARDENING, SECURITY & ERROR RECOVERY

### 1. Fundamentals
Production-grade software systems must handle unexpected edge cases, invalid inputs, network drops, and potential security vulnerabilities gracefully without crashing or corrupting data.

### 2. Why We Use This in This Project
Medical software must adhere to strict reliability standards. A system crash during emergency telemedicine could disrupt patient care. We hardened our codebase against common failure modes.

### 3. Exact Repository Implementation
- Path sanitization & disk margins: `receiver/image_receiver.py` (Lines 42-65)
- Framing safety limits: `shared/protocol.py` (Lines 51-54)
- Reachability probing & retry backoff: `sender/tcp_client.py` (Lines 60-115)
- Centralized configuration: `common/config.py` (Lines 1-135)

### 4. Hardened Security & Reliability Features
```
┌─────────────────────────────────────────────────────────────┐
│                 DEFENSE-IN-DEPTH ARCHITECTURE               │
├─────────────────────────────────────────────────────────────┤
│ 1. Memory Allocation Bounds (DoS Protection)                │
│    - MAX_METADATA_SIZE = 64 KB                              │
│    - MAX_IMAGE_PAYLOAD_SIZE = 50 MB                         │
├─────────────────────────────────────────────────────────────┤
│ 2. Filesystem Security (Path Traversal Protection)          │
│    - Strips '..', '/', and '\' from image_id & exp_id       │
│    - Resolves target path and enforces directory containment│
├─────────────────────────────────────────────────────────────┤
│ 3. Storage Safety Margin (Disk Out-of-Space Protection)     │
│    - Enforces MIN_FREE_DISK_BYTES = 50 MB safety margin     │
├─────────────────────────────────────────────────────────────┤
│ 4. Network Resilience (Socket Reconnection & Probing)       │
│    - Pre-flight RTT reachability probing                    │
│    - Exponential backoff retry loop (1s, 2s, 4s)            │
└─────────────────────────────────────────────────────────────┘
```

### 5. Input, Processing, and Output
- **Input**: Unsanitized network inputs and socket connection attempts.
- **Processing**: Header length verification $\rightarrow$ Path character filtering $\rightarrow$ Disk capacity check $\rightarrow$ Connection timeout handling.
- **Output**: Safe system execution, structured error logging, and clean recovery.

### 6. Module Interconnections
These safeguards operate across `shared/protocol.py`, `receiver/image_receiver.py`, and `sender/tcp_client.py`.

### 7. Mathematical Foundation
Exponential backoff retry delay:
$$t_{\text{wait}}(k) = \min(t_{\text{initial}} \cdot 2^{k-1}, t_{\text{max}})$$
For retry attempt $k \in \{1, 2, 3\}$, wait times are $1.0\text{ s}, 2.0\text{ s}, 4.0\text{ s}$, which prevents network congestion during server recovery.

### 8. Concrete Project Data & Examples
- **Path Traversal Defense**: If a malicious client sends `image_id = "../../../../windows/system32/cmd.exe"`, the receiver sanitizes the string to `________windows_system32_cmd.exe` and verifies that the destination path remains inside `data/received/`, rejecting the request if it escapes.
- **Disk Protection**: If the receiver drive has less than $50 \text{ MB}$ of free space, `save_image()` raises an `OSError` before writing, preventing filesystem exhaustion.

### 9. How to State This in the Viva
> *"We implemented several production-grade safeguards: First, strict buffer limits (64 KB metadata, 50 MB payload) to prevent memory exhaustion attacks. Second, path sanitization to eliminate path traversal vulnerabilities. Third, a 50 MB disk margin check to prevent disk exhaustion. Fourth, exponential backoff and pre-flight RTT probing to ensure network resilience. All errors are caught and logged without crashing the application."*

### 10. Viva Questions & Model Answers
- **Q1: How does your system handle corrupted JPEG bytes received over the network?**
  *Answer*: The receiver verifies the SHA-256 checksum before decoding. If a bit is corrupted, the hash check fails and the file is rejected. If an uncorrupted but invalid JPEG is received, OpenCV/PIL decoding errors are caught in `segmentation.py`, which logs the error and returns a clean failure status without crashing.
- **Q2: What prevents the sender GUI from freezing during large batch transfers?**
  *Answer*: All network communication and batch processing run inside dedicated `QThread` background workers (`BatchRunnerWorker`). The worker communicates with the PySide6 GUI purely through asynchronous Qt signals (`experiment_completed`, `progress_updated`), keeping the user interface responsive at 60 FPS.

---

# MODULE 12: VIVA VOCE MASTER DEFENSE PLAYBOOK (TOP 20 QUESTIONS & ANSWERS)

### Q1. What is the core novelty of this project?
**Answer**: Many existing works evaluate medical image compression and deep learning segmentation as separate topics, or simulate network behavior purely using artificial delays in software. Our novelty lies in building an **end-to-end, physical dual-laptop testbed** that combines kernel-level traffic control (`tc/netem`), real-time throughput-adaptive JPEG compression, a lightweight binary wire protocol with SHA-256 verification, and a frozen U-Net segmentation model evaluated against pixel-perfect ground truth on the benchmark BUSI dataset.

### Q2. Walk me through the step-by-step life of an ultrasound image from selection to segmentation.
**Answer**:
1. The user selects an ultrasound scan on the Ubuntu sender GUI.
2. The sender determines the compression quality (fixed or via the adaptive controller's EWMA throughput estimate).
3. The image is compressed in memory, and its PSNR, SSIM, and SHA-256 hash are computed.
4. The payload is packed into a `NAC1` binary frame (14-byte header, metadata JSON, raw image bytes) and streamed over the physical Ethernet link.
5. The Windows receiver's multi-threaded daemon reads the frame using `recv_exact`, validates header bounds, and verifies the SHA-256 checksum.
6. The image is saved to `data/received/` with a sanitized filename.
7. The frozen U-Net model preprocesses the image to $256 \times 256 \times 1$ and runs GPU inference under `torch.no_grad()`.
8. The predicted mask is evaluated against the combined ground-truth mask to compute Dice and IoU scores.
9. An acknowledgment frame (`NACK`) is sent back to the sender.
10. Both GUIs update their live displays, and the complete record is logged to SQLite and CSV.

### Q3. Why did you freeze the U-Net weights on the receiver node?
**Answer**: If the segmentation model continued training or updating weights during experiments, changes in segmentation performance could be caused by model weight updates rather than compression artifacts. Freezing the model ensures that the segmentation model remains an independent, constant measuring instrument, isolating image compression as the sole experimental variable.

### Q4. Explain the difference between PSNR and SSIM.
**Answer**:
- **PSNR** is based on Mean Squared Error (MSE). It measures absolute pixel-by-pixel intensity differences on a logarithmic scale. It does not account for human visual perception or spatial relationships between neighboring pixels.
- **SSIM** evaluates image degradation across three perceptually relevant factors: luminance correlation, contrast correlation, and structural correlation. It compares local pixel neighborhoods, making it a better predictor of whether edge boundaries and anatomical textures were preserved for downstream convolutional neural networks.

### Q5. What is the mathematical relationship between Dice and IoU?
**Answer**:
$$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}} \quad \text{and} \quad \text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$
Because the denominator $(2 - \text{Dice}) \ge 1.0$, Dice is always greater than or equal to IoU for any non-empty region.

### Q6. How does your system handle healthy normal ultrasound scans with no lesions?
**Answer**: Normal scans have no lesion annotations. Standard Dice formulas produce $\frac{0}{0}$ on empty masks. We implemented an explicit Empty Mask Policy:
- If both the ground truth and the prediction are empty ($|G| = 0$ and $|P| = 0$), it represents a **True Negative**, yielding $\text{Dice} = 1.0$ and $\text{IoU} = 1.0$.
- If the model predicts any lesion pixels on an empty ground truth ($|G| = 0$ and $|P| > 0$), it represents a **False Positive**, yielding $\text{Dice} = 0.0$ and $\text{IoU} = 0.0$.
In our test split baseline of 20 normal scans, the model achieved $1.0$ on 16 scans and $0.0$ on 4 scans, resulting in an 80% mean score.

### Q7. What are the trainable parameters and training specs of your U-Net model?
**Answer**:
- **Architecture**: 4 downsampling stages (DoubleConv + MaxPool), 512-channel bottleneck, 4 upsampling stages (ConvTranspose + Skip Connections), and a 1x1 Conv head.
- **Parameters**: Exactly **7,762,465** trainable parameters.
- **Training**: 25 epochs in 453.71 seconds on an NVIDIA RTX 3050 GPU using AdamW ($\text{lr} = 0.001$, weight decay = $10^{-4}$) and Cosine Annealing.
- **Best Checkpoint**: Epoch 21 with validation Dice of 0.6056, IoU of 0.5287, and validation loss of 0.3830.

### Q8. What baseline segmentation results did your model achieve on the held-out test split?
**Answer**: Evaluated on all 117 uncompressed test scans:
- **Overall**: Mean Dice = **0.6552** (Median = **0.8091**), Mean IoU = **0.5733** (Median = **0.6793**).
- **Lesion cases only (97 scans)**: Mean Dice = **0.6254** (Median = **0.7787**), Mean IoU = **0.5266** (Median = **0.6376**).
- **Benign cohort (66 scans)**: Mean Dice = **0.6414**, Mean IoU = **0.5462**.
- **Malignant cohort (31 scans)**: Mean Dice = **0.5913**, Mean IoU = **0.4848**.
- **Normal cohort (20 scans)**: Mean Dice = **0.8000**, Mean IoU = **0.8000**.
- **Average GPU inference latency**: **6.95 ms per image**.

### Q9. Why did you use Linux `tc/netem` on the sender instead of software sleep functions?
**Answer**: In-memory software sleep (`time.sleep`) bypasses the operating system network stack, socket buffers, and physical network interfaces. Linux `tc/netem` operates inside the Linux kernel on the physical network interface (`eth0`), deterministically throttling bandwidth and injecting latency and packet drops. This subjects the TCP stream to real packet queuing, congestion control adjustments, and retransmission delays.

### Q10. How does the custom binary framing protocol prevent TCP stream fragmentation issues?
**Answer**: TCP is a stream-oriented transport protocol without message boundaries; calling `recv(N)` can return partial byte chunks. We implemented a `recv_exact` loop that reads from the socket until the exact requested byte count is received. In addition, the 14-byte `NAC1` binary header explicitly specifies metadata length and image payload length, allowing unambiguous, deterministic frame demaracation.

### Q11. How does the adaptive rate controller choose the JPEG quality factor?
**Answer**: The controller calculates the instantaneous throughput of each transmission and smooths the estimate using an Exponentially Weighted Moving Average (EWMA) with $\alpha = 0.3$:
$$\text{EWMA}_t = 0.3 \cdot R_t + 0.7 \cdot \text{EWMA}_{t-1}$$
The smoothed bandwidth is mapped to quality tiers:
- $\ge 10\text{ Mbps} \implies \text{Q90}$
- $2\text{ to } 10\text{ Mbps} \implies \text{Q70}$
- $512\text{ Kbps to } 2\text{ Mbps} \implies \text{Q50}$
- $< 512\text{ Kbps} \implies \text{Q30}$

### Q12. How does the system guarantee that data was not corrupted during transmission?
**Answer**: Before transmission, the sender calculates a cryptographic SHA-256 hash of the compressed byte array and includes it in the metadata JSON. Upon receiving the payload, the receiver independently computes the SHA-256 hash over the received bytes. If the two digests match, an `ACK_OK` is returned; if they differ, the receiver emits `ACK_CHECKSUM_MISMATCH` and aborts processing.

### Q13. Why are ground-truth masks never compressed with JPEG?
**Answer**: Ground-truth masks represent clinical reference boundaries. Applying lossy JPEG compression to a binary mask introduces ringing artifacts and gradient blur across boundaries, corrupting the reference annotations. The compression engine explicitly validates filenames and raises an error if a mask file is passed for compression.

### Q14. What security measures are implemented against malicious payloads?
**Answer**:
1. **Memory bounds**: Metadata is capped at 64 KB and payload at 50 MB to prevent memory allocation attacks.
2. **Path traversal defense**: `image_id` and `experiment_id` strings are stripped of directory traversal characters (`..`, `/`, `\`), and the resolved destination path is verified to remain inside the target storage directory.
3. **Disk protection**: The receiver checks for a 50 MB free disk space safety margin before writing files.

### Q15. Why does the receiver use multi-threading for the TCP server?
**Answer**: The TCP server uses a dedicated listening thread with a 1.0-second socket timeout to accept incoming connections without blocking the main application. Each accepted client connection is handled in a separate thread, allowing the server to shut down cleanly and process images while keeping the desktop GUI responsive.

### Q16. How does your system combine multi-lesion masks in the BUSI dataset?
**Answer**: Scans with multiple lesions have separate mask files (e.g., `benign (10)_mask.png` and `benign (10)_mask_1.png`). Our dataset manager scans for all associated masks and combines them using a pixel-wise logical OR operation ($\max(M_1, M_2)$) resized with nearest-neighbor interpolation. This ensures that all annotated lesion regions are represented in the evaluation.

### Q17. How does the system handle an unexpected receiver shutdown?
**Answer**: The sender's `TCPClient` sets configurable socket timeouts on all network calls. If the receiver terminates unexpectedly, the sender detects a socket error (`ConnectionResetError` or timeout), logs the failure to SQLite, closes the socket, and displays an error message on the GUI without crashing.

### Q18. How do you distinguish theoretical wire delay from measured transmission time?
**Answer**:
- **Theoretical wire delay** is calculated as $T_{\text{theo}} = \frac{\text{Bytes} \times 8}{\text{Bandwidth}}$. It represents the minimum time required to push raw bits across an idealized link.
- **Measured transmission time** is recorded using hardware monotonic timers (`time.perf_counter()`) across the physical socket send loop. It captures real-world factors including socket buffer copy time, NIC packetization, RTT latency, and TCP retransmissions. Both metrics are logged in our database.

### Q19. What are the key visual artifacts introduced by JPEG compression at Q30?
**Answer**: At Q30, aggressive quantization of high-frequency DCT coefficients causes $8 \times 8$ block boundary artifacts (grid patterns across smooth tissue areas), ringing artifacts around sharp acoustic borders, and attenuation of fine speckle textures. These artifacts can blur lesion boundaries, leading to slightly lower Dice scores on subtle malignant tumors.

### Q20. If you had 6 more months to extend this research, what would you add?
**Answer**:
1. Implement modern neural image compression (e.g., learned autoencoder compression with hyperprior models) trained with a joint rate-distortion-segmentation loss.
2. Extend the network controller to simulate dynamic cellular handovers with time-varying bandwidth traces recorded from moving emergency vehicles.
3. Implement Region-of-Interest (ROI) adaptive compression, where an ultra-lightweight detection model identifies lesion bounding boxes to compress them losslessly while aggressively compressing background tissue.
