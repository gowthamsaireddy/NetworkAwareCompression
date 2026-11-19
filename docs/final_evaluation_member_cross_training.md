# Final Evaluation Cross-Training Guide: Multi-Disciplinary Defense
## Project: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

---

# SECTION 1: THE CROSS-EXAMINATION STRATEGY

### 1.1 Why Cross-Training Determines Your Grade
In final-year project evaluations, examination committees intentionally test team cohesion by directing questions **outside each student's declared primary area**.
- If Member 1 (DIP) cannot answer how the image is transmitted across TCP sockets, the committee assumes the team operated in isolated silos.
- If Member 2 (Deep Learning) cannot explain why JPEG quantization causes blocking artifacts, the committee assumes models were used as black boxes.
- If Member 3 (Networks) cannot explain why U-Net skip connections preserve tumor boundaries, the committee assumes the network pipeline was built without understanding medical data requirements.
- If Member 4 (Integration/Security) cannot explain the mathematical difference between Dice and IoU, the committee assumes integration was superficial.

### 1.2 The Two-Step Defense Protocol: "Answer & Bridge"
When a faculty member asks you a question outside your primary specialty, **never say "That was not my part" or "Member X did that."**
Follow the **Two-Step Defense Protocol**:
1. **Step 1 (The Confident Core Answer):** Deliver a crisp, 15-to-30 second technically accurate summary of the underlying engineering principle.
2. **Step 2 (The Seamless Bridge):** Conclude by inviting the domain specialist to elaborate on deeper architectural nuances if the panel desires.

> **Formula:** *"While my teammate [Member Name] directly implemented the low-level module, the core engineering principle is [concise technical answer]. In our project specifically, we achieved this by [repository fact]. [Member Name] can elaborate on the exact implementation details."*

---

# SECTION 2: MEMBER 1 CROSS-TRAINING DEFENSE GUIDE
### (Primary: DIP, BUSI Dataset, Standard JPEG Compression, Image Quality Metrics)

As Member 1, examiners will try to catch you off-guard on **Deep Learning**, **Computer Networks**, and **Security**.

---

### Cross-Defense 1: Member 1 Answering Deep Learning & U-Net
- **Faculty Trap Question:** *"You talked about image processing and resizing. How does the receiver's AI actually find the lesion once it receives your compressed image? What is U-Net?"*
- **Your Model Answer:**
  "Our receiver workstation runs a 4-stage biomedical U-Net comprising 7.76 million parameters. When the resized $256 \times 256$ image enters the network, an encoder path extracts high-level semantic context through repeated $3 \times 3$ convolutions and max pooling. Lateral skip connections copy sharp, fine-grained spatial feature maps from the encoder directly across to the decoder, restoring spatial localization lost during downsampling. A final $1 \times 1$ convolution with a sigmoid activation generates pixel-wise lesion probabilities thresholded at $\tau = 0.5$. Member 2 also optimized this network using the Multi-Gradient Descent Algorithm (MGDA) to eliminate gradient conflict between BCE and Dice loss, achieving a validation Dice score of 0.6581."
- **Key Numbers to Recite:** $7,762,465$ parameters, $31.2\text{ MB}$, Sigmoid threshold $\tau = 0.5$, Validation Dice $0.6581$.

---

### Cross-Defense 2: Member 1 Answering Computer Networks & Sockets
- **Faculty Trap Question:** *"You compressed the image to 48 KB. Why did you use TCP instead of UDP to transmit it? UDP is much faster!"*
- **Your Model Answer:**
  "While UDP has lower header overhead, it provides an unreliable datagram service with zero loss guarantees. In medical ultrasound transmission, packet loss under UDP would corrupt the JPEG Huffman entropy stream; dropping even a single packet containing variable-length codes desynchronizes the entropy decoder and destroys the entire reconstructed scan. TCP guarantees reliable, in-order byte delivery through sequence numbers, flow control, and retransmissions. Member 3 implemented a custom 14-byte binary framing protocol (`b"NAC1"`) over TCP, achieving a physical transmission time of just 4.55 milliseconds on Gigabit Ethernet."
- **Key Numbers to Recite:** 14-byte header (`!4sHII`), 16-byte ACK (`!4sIQ`), $4.55\text{ ms}$ wire time, $86.2\text{ Mbps}$ throughput.

---

### Cross-Defense 3: Member 1 Answering Security & Verification
- **Faculty Trap Question:** *"How does your system guarantee that the received image wasn't corrupted in transit or tampered with by a malicious attacker?"*
- **Your Model Answer:**
  "Before dispatch, our sender computes a 256-bit SHA-256 cryptographic digest across the compressed payload and transmits it in the frame metadata. Upon receiving the payload, the receiver recalculates the hash before decoding. If a single bit was flipped during transmission or modified in transit, the digests do not match; the receiver immediately rejects the frame, aborts segmentation, and returns an error ACK. Member 4 verified that our SHA-256 engine achieves a 100% bit-flip detection rate with a mean detection time of just 0.34 milliseconds."
- **Key Numbers to Recite:** 256-bit hash, MTTD $0.34\text{ ms}$, Throughput $>380\text{ MB/s}$.

---

# SECTION 3: MEMBER 2 CROSS-TRAINING DEFENSE GUIDE
### (Primary: Deep Learning, U-Net, MGDA Optimization, ROI Detector, Adaptive Compression)

As Member 2, examiners will try to catch you off-guard on **DCT Quantization**, **Networking Sockets**, and **Database/Storage**.

---

### Cross-Defense 1: Member 2 Answering DIP & Standard JPEG Compression
- **Faculty Trap Question:** *"You claim aggressive compression degrades your U-Net's performance. Mathematically, why does JPEG compression create blocky grid lines at Q30?"*
- **Your Model Answer:**
  "Standard JPEG partitions the image into independent $8 \times 8$ pixel blocks and applies the 2D Discrete Cosine Transform. Quantization is the only lossy step: DCT coefficients are divided by a scaled quantization table and rounded to integers. At Q30, the scaling factor multiplies divisors by nearly $1.67\times$, rounding virtually all high-frequency AC coefficients to zero. Each $8 \times 8$ block is reduced to its flat DC average. Discontinuities between the independent DC averages of adjacent blocks create visible $8 \times 8$ rectilinear grid boundaries that falsely trigger the convolutional edge kernels in our U-Net."
- **Key Formulas to Write:** $F_Q(u, v) = \text{round}\left( \frac{F(u, v)}{Q(u, v)} \right)$, $S = \frac{5000}{Q}$.

---

### Cross-Defense 2: Member 2 Answering Networking & Linux `tc netem`
- **Faculty Trap Question:** *"How did you prove that your region-adaptive compression actually helps under bad network conditions? Did you just test on Wi-Fi?"*
- **Your Model Answer:**
  "No, we deployed across physical hardware and tested beyond ideal local links by using Linux Traffic Control (`tc`) and the `netem` kernel module on our Ubuntu sender. Member 3 configured six clinical profiles, including a 384 Kbps Rural Telemedicine Link with 120 ms latency and 2.5% packet loss. Across that rural link, an uncompressed 455 KB image requires 9.61 seconds to transmit, whereas our 48 KB region-adaptive image transmits in just 1.14 seconds—an 8.4-fold speedup that prevents TCP bufferbloat and transfer timeouts."
- **Key Numbers to Recite:** Rural Link: $384\text{ Kbps}$, $120\text{ ms}$ delay, $2.5\%$ loss; Transmission speedup: $9.61\text{ s} \to 1.14\text{ s}$ ($8.4\times$).

---

### Cross-Defense 3: Member 2 Answering System Integration & SQLite
- **Faculty Trap Question:** *"Where are your segmentation scores and inference latencies stored? What prevents database crashes during rapid batch runs?"*
- **Your Model Answer:**
  "Every experimental result is logged into an ACID-compliant 42-column SQLite database at `experiments/experiments.sqlite`. To prevent database lockups during concurrent GUI queries and incoming socket writes, Member 4 configured Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) with a 10-second busy timeout. This allows the GUI results table to read records concurrently while the server background thread commits new inference trials. Our database currently contains 28 fully verified physical experimental records."
- **Key Facts to Recite:** 42 columns, WAL mode (`PRAGMA journal_mode=WAL;`), 28 verified physical trials.

---

# SECTION 4: MEMBER 3 CROSS-TRAINING DEFENSE GUIDE
### (Primary: Computer Networks, TCP Sockets, Binary Framing, Telemetry, Two-Laptop Deployment)

As Member 3, examiners will try to catch you off-guard on **Digital Image Processing**, **Deep Learning Architectures**, and **Security Hardening**.

---

### Cross-Defense 1: Member 3 Answering DIP & Preprocessing
- **Faculty Trap Question:** *"You are sending images across the network. How does the sender prepare the raw ultrasound scans before your socket code ever touches them?"*
- **Your Model Answer:**
  "The sender ingests scans from the 780-image BUSI dataset. In `common/dataset.py`, Member 1 implemented a multi-mask reconciliation pipeline: when scans contain multiple tumor foci, element-wise maximum (`np.maximum`) merges them into a unified binary ground truth. The dataset is partitioned into 70% train (546), 15% validation (117), and 15% test (117) using stratified sampling fixed at seed 42 to prevent data leakage. Images are resized using bilinear interpolation to preserve acoustic gradients, while masks strictly use nearest-neighbor interpolation to prevent fractional label corruption."
- **Key Facts to Recite:** 780 images (437 benign, 210 malignant, 133 normal), 70/15/15 split, seed 42, bilinear for images vs nearest-neighbor for masks.

---

### Cross-Defense 2: Member 3 Answering Deep Learning & ROI Detection
- **Faculty Trap Question:** *"Your network transmits an 'adaptive' payload. Who decides what part of the image is the lesion, and how does that happen at the edge?"*
- **Your Model Answer:**
  "To detect the lesion at the edge without a heavy GPU, Member 2 engineered a compact ROI Detector in `models/roi_detector.py`. It has only 482,449 parameters—over 16 times smaller than our clinical U-Net—and executes on our Ubuntu laptop CPU in just 3.82 milliseconds. It predicts a candidate lesion mask, which is expanded using elliptical morphological dilation with a 15-pixel margin and smoothed with Gaussian feathering ($\sigma = 5.0$). This allows the compressor to preserve the tumor at Q90 while compressing the background at Q20 before my socket transport streams the resulting 48 KB payload."
- **Key Numbers to Recite:** $482,449$ parameters, $1.96\text{ MB}$, $3.82\text{ ms}$ CPU latency, 15 px margin, $\sigma = 5.0$.

---

### Cross-Defense 3: Member 3 Answering Security & Attack Vectors
- **Faculty Trap Question:** *"What happens if an attacker connects to your TCP listening port 5000 and floods it with a 10 GB stream of junk data?"*
- **Your Model Answer:**
  "Our architecture implements application-layer Denial-of-Service (DoS) protection in `receiver/tcp_server.py`. Immediately after parsing the 14-byte `NAC1` frame header, the server inspects the 4-byte payload length field. If the advertised length exceeds our hard ceiling of 50 MB ($52,428,800\text{ bytes}$), the socket connection is dropped instantly before calling `bytearray()`. Memory is never allocated for the oversized payload, neutralizing memory exhaustion attacks before they reach RAM."
- **Key Numbers to Recite:** 50 MB hard payload ceiling, pre-allocation header check.

---

# SECTION 5: MEMBER 4 CROSS-TRAINING DEFENSE GUIDE
### (Primary: System Integration, GUI Architecture, Database, Security, Research Synthesis)

As Member 4, examiners will try to catch you off-guard on **DCT Quantization Mathematics**, **U-Net Multi-Objective Loss**, and **TCP Transport Physics**.

---

### Cross-Defense 1: Member 4 Answering DIP & Image Metrics
- **Faculty Trap Question:** *"You logged PSNR and SSIM in your database. Mathematically, what is the exact formula for PSNR, and why did you also need SSIM?"*
- **Your Model Answer:**
  "PSNR is defined as $\text{PSNR} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}} \right)$ for 8-bit images. While PSNR measures absolute pixel error, it is a global spatial average that treats all pixels equally; an image with a heavily blurred lesion can still maintain a deceptively high global PSNR if the background is preserved. That is why we also calculate SSIM, which models perceived luminance, contrast, and structural correlation using Gaussian sliding windows, as well as a specialized ROI-PSNR metric that isolates distortion exclusively within the lesion boundary."
- **Key Formulas to Write:** $\text{PSNR} = 10 \cdot \log_{10}\left( \frac{65025}{\text{MSE}} \right)$, $\text{MSE} = \frac{1}{MN} \sum (I - K)^2$.

---

### Cross-Defense 2: Member 4 Answering MGDA Optimization
- **Faculty Trap Question:** *"What is the core mathematical problem that MGDA solves during U-Net training? Why not just use standard backpropagation?"*
- **Your Model Answer:**
  "Standard backpropagation with scalar combined loss $\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$ fails when task gradients conflict ($\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0$). Because BCE optimizes global pixel accuracy while Dice optimizes foreground overlap, their gradient vectors frequently pull parameters in opposing directions, causing destructive gradient interference. MGDA formulates training as multi-objective optimization, finding the analytical Frank-Wolfe minimum norm vector on the Pareto frontier: $\alpha^* = \text{clip}\left(\frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1\right)$. This dynamic reweighting eliminated gradient conflict and increased our validation Dice score from 0.6056 to 0.6581."
- **Key Formula to Write:** $\alpha^* = \text{clip}\left( \frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1 \right)$.

---

### Cross-Defense 3: Member 4 Answering TCP Wire Telemetry
- **Faculty Trap Question:** *"On your results screen, measured throughput shows 86.2 Mbps across Gigabit Ethernet. Why didn't it achieve 1000 Mbps?"*
- **Your Model Answer:**
  "Throughput is constrained by TCP Slow Start (RFC 5681) on short data bursts. Our total adaptive frame payload is 49,007 bytes, requiring only about 33 TCP segments (MSS = 1460 bytes). A new TCP connection starts with an initial congestion window of approximately 10 segments. The entire 49 KB transmission finishes in just two round-trip times during the initial slow-start window ramp-up, long before the connection can expand its congestion window to saturate the 1 Gbps physical line rate. Member 3 clocked this wire time strictly around `sock.sendall()` using monotonic `time.perf_counter()`, recording an exact wire time of 4.55 milliseconds."
- **Key Facts to Recite:** 49,007 bytes payload, ~33 TCP segments, initial congestion window ($IW \approx 10$), wire time $4.55\text{ ms}$, throughput $86,214.85\text{ Kbps}$.

---

# SECTION 6: THE COMPREHENSIVE CROSS-TRAINING MATRIX

Use this rapid lookup table during final mock rehearsals:

| Technical Topic | Primary Specialist | Member 1 Defense Point | Member 2 Defense Point | Member 3 Defense Point | Member 4 Defense Point |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BUSI Dataset & Splits** | **Member 1** | Primary owner: 780 scans, seed 42, 70/15/15 split. | Uses `splits['train']` strictly; test set quarantined. | Transmits test scans deterministically across sockets. | Logs `image_id` and category into SQLite provenance tables. |
| **Multi-Mask Union** | **Member 1** | `np.maximum(m1, m2)` prevents overflow; set union. | Trains models against unified multi-focal masks. | Transmits single unified case metadata frame. | Verifies ground truth matching during receiver scoring. |
| **JPEG Quantization** | **Member 1** | Sole lossy step; $F_Q = \text{round}(F/Q)$; scales by $S$. | Explains blocking artifacts that trigger false CNN edges. | Notes JPEG payload byte sizes stream over TCP. | Logs quality factors and size reduction percentages. |
| **U-Net Architecture** | **Member 2** | Prepares $256 \times 256$ inputs via bilinear resizing. | Primary owner: 7.76M params, skip connections, sigmoid. | Receives payload, feeds decoded buffer to U-Net. | Embeds U-Net in receiver inference pipeline; logs latency. |
| **MGDA Optimization** | **Member 2** | Understands BCE/Dice loss trade-offs in pixel accuracy. | Primary owner: Frank-Wolfe $\alpha^*$ closed form, +5.25% gain. | Knows MGDA weights run frozen during transmission tests. | Logs model checkpoint versions and validation gains. |
| **ROI Edge Detector** | **Member 2** | Provides mask for elliptical margin dilation. | Primary owner: 482K params, 1.96 MB, 3.82 ms latency. | Understands sender CPU runs ROI detector before socket write. | Benchmarks edge latency vs hospital receiver latency. |
| **TCP Sockets & Framing** | **Member 3** | Formats image bytes before passing to socket client. | Understands why TCP guarantees zero Huffman corruption. | Primary owner: 14B `NAC1`, 16B `NACK`, `recv_exact`. | Integrates client socket calls within batch orchestrator. |
| **Monotonic Wire Telemetry**| **Member 3** | Cites wire times in relation to compressed byte sizes. | Relates throughput to inference latency budgets. | Primary owner: `time.perf_counter()`, 86.2 Mbps wire rate. | Implemented telemetry backfilling (`_backfill_missing_telemetry`). |
| **Linux `tc netem`** | **Member 3** | Notes compression benefits on 384 Kbps rural links. | Cites 8.4× speedup preventing edge timeouts. | Primary owner: qdisc rules, 6 clinical profiles. | Includes profile parameters in 42-column SQLite schema. |
| **SHA-256 Verification** | **Member 4** | Generates payload bytes hashed by SHA-256. | Understands integrity check precedes image decoding. | Receives digest in JSON; aborts if mismatch. | Primary owner: MTTD 0.34 ms, 100% bit-flip detection. |
| **42-Column SQLite DB** | **Member 4** | Reviews compression and distortion metric columns. | Reviews Dice, IoU, and lesion area columns. | Reviews wire bytes, tx time, and throughput columns. | Primary owner: Schema design, WAL mode, dynamic migration. |
| **Controlled Batch Engine** | **Member 4** | Configures selected images and quality lists. | Validates model evaluation modes (model vs ground truth).| Monitors socket connections during batch execution. | Primary owner: State machine, pause/resume/cancel. |
