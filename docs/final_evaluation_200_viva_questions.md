# Comprehensive Viva Examination Question Bank (208 Questions)
## Project: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation

---

# CATEGORY 1: PROJECT MOTIVATION, CLINICAL PROBLEM & TELEMEDICINE ARCHITECTURE

#### Q1: What is the clinical motivation behind this project?
- **Direct Answer:** To enable fast, reliable transmission of breast ultrasound images from rural edge clinics to metropolitan hospitals over bandwidth-constrained networks without sacrificing automated lesion segmentation accuracy.
- **Detailed Explanation:** Breast cancer is a major global health threat. Early detection via ultrasound is radiation-free and effective, but expert radiologists are concentrated in urban hospitals. Rural clinics rely on low-bandwidth, high-latency 3G, 4G, or satellite links. Transmitting large uncompressed images causes long delays, while aggressive uniform compression introduces blocky artifacts that destroy lesion boundaries. Our system resolves this trade-off using region-adaptive compression.
- **Code Reference:** `docs/research_summary_report.md`
- **Formula:** $\text{Latency} = \frac{\text{Payload Bits}}{\text{Bandwidth}} + \text{RTT}$.

#### Q2: What is B-mode ultrasound imaging?
- **Direct Answer:** Brightness-mode (B-mode) ultrasound is a 2D cross-sectional diagnostic imaging technique where pixel brightness corresponds to the amplitude of acoustic echoes reflected from internal tissues.
- **Detailed Explanation:** A transducer converts electrical pulses into high-frequency sound waves ($1\text{–}18\text{ MHz}$) that propagate into breast tissue. When waves hit acoustic impedance boundaries, echoes reflect back. Time delay determines depth, and echo intensity determines 8-bit grayscale brightness.
- **Code Reference:** `common/dataset.py:BUSIDataset`

#### Q3: Why is lossy compression traditionally avoided or controversial in medical imaging?
- **Direct Answer:** Because lossy compression permanently discards high-frequency pixel information, which can blur subtle lesion margins, erase microcalcifications, or create artificial blocky edges that lead to clinical misdiagnosis.
- **Detailed Explanation:** Regulatory bodies (FDA, ACR) mandate that any lossy compression used in primary diagnosis or automated triage must be scientifically proven not to impair diagnostic efficacy. Our project provides this empirical proof by demonstrating zero degradation in U-Net segmentation Dice scores ($0.8664$ vs $0.8664$).
- **Code Reference:** `docs/limitations.md`

#### Q4: What is the primary objective of Region-Adaptive Compression?
- **Direct Answer:** To preserve the diagnostic Region of Interest (lesion and surrounding parenchymal margin) at pristine high quality (Q90) while aggressively compressing non-diagnostic background tissue at low quality (Q20).
- **Detailed Explanation:** In breast ultrasound, over 80–90% of image pixels represent surrounding fat, muscle, or acoustic shadows. By concentrating bit budget strictly on the clinically relevant lesion, total byte size is reduced by ~38% without compromising diagnostic AI segmentation.
- **Code Reference:** `sender/adaptive_compressor.py:RegionAdaptiveCompressor`

#### Q5: What is the difference between client-server and peer-to-peer topologies in telemedicine?
- **Direct Answer:** Telemedicine follows a client-server architecture where multiple distributed edge clinics (clients/senders) transmit diagnostic data to a central hospital PACS server (receiver) for archival and high-performance AI inference.
- **Detailed Explanation:** Client-server topology ensures centralized database provenance, secure audit logging, and access to heavy GPU infrastructure, which rural edge nodes lack.
- **Code Reference:** `sender/tcp_client.py`, `receiver/tcp_server.py`

#### Q6: How does your system define a "Region of Interest" (ROI)?
- **Direct Answer:** The lesion area detected by our edge AI model or ground truth, expanded radially by a configurable margin (default: 15 pixels) to protect the peritumoral transition zone.
- **Detailed Explanation:** Peritumoral tissue contains critical diagnostic information such as acoustic spiculation, micro-lobulation, and desmoplastic reactions. Expanding the lesion mask guarantees this boundary is not clipped by background compression.
- **Code Reference:** `sender/adaptive_compressor.py:112-128`
- **Formula:** $M_{\text{dilated}} = M \oplus B_{\text{ellipse}}(r=15)$.

#### Q7: Why are two separate computers used instead of running sender and receiver on one laptop?
- **Direct Answer:** To demonstrate true physical networking across distinct operating systems, testing socket fragmentation, kernel queuing, MTU limits, and wire clocking on real physical hardware.
- **Detailed Explanation:** Running both processes on localhost (`127.0.0.1`) bypasses physical network cards, kernel egress queues, physical serialization delays, and operating system firewall rules, masking real-world networking bottlenecks.
- **Code Reference:** `docs/two_laptop_setup_guide.md`

#### Q8: What are the two distinct compression modes available in the sender GUI?
- **Direct Answer:** Option A: Normal Compression (uniform standard JPEG) and Option B: ML-Based Region-Adaptive Compression.
- **Detailed Explanation:** Both modes are independently selectable via dedicated radio buttons in `ui/sender/compression_view.py`. Option A maintains the legacy baseline, while Option B activates the edge ROI detector, morphological dilation, and dual-rate blending engine.
- **Code Reference:** `ui/sender/compression_view.py:95-120`

---

# CATEGORY 2: BUSI DATASET ORGANIZATION, DEMOGRAPHICS & MULTI-MASK UNION

#### Q9: What does BUSI stand for?
- **Direct Answer:** Breast Ultrasound Images dataset.
- **Detailed Explanation:** A public benchmark medical image dataset created in 2018 by Al-Dhabyani et al. at Baheya Hospital for Early Detection & Treatment of Women's Cancer, Cairo, Egypt.
- **Code Reference:** `common/dataset.py:15`

#### Q10: How many total images and categories are in the BUSI dataset?
- **Direct Answer:** Exactly 780 images across 3 categories: 437 Benign, 210 Malignant, and 133 Normal.
- **Detailed Explanation:** Benign cases exhibit circumscribed oval/round hypoechoic masses. Malignant cases exhibit irregular, spiculated hypoechoic lesions with posterior acoustic shadowing. Normal cases depict healthy parenchymal tissue without focal masses.
- **Code Reference:** `common/dataset.py:40-48`
- **Formula:** $N_{\text{total}} = 437 + 210 + 133 = 780$.

#### Q11: What patient demographic characteristics are represented in BUSI?
- **Direct Answer:** Scans were collected from 600 female patients aged between 25 and 75 years using a LOGIQ E9 ultrasound system.
- **Detailed Explanation:** The wide age range captures varying breast tissue densities, from dense fibro-glandular tissue common in younger women to fatty involution in post-menopausal women.
- **Code Reference:** `docs/research_summary_report.md`

#### Q12: Why are there multiple mask files for certain images in BUSI?
- **Direct Answer:** Because certain patients presented multifocal or multicentric lesions, resulting in separate radiologist annotations for each distinct tumor focus.
- **Detailed Explanation:** For instance, `benign (100).png` has `benign (100)_mask.png` and `benign (100)_mask_1.png`. If only the primary mask is read, secondary tumor foci are omitted from ground truth.
- **Code Reference:** `common/dataset.py:50-58`

#### Q13: How does your code merge multiple masks for a single scan?
- **Direct Answer:** Using an element-wise maximum operation (`np.maximum`), which performs a boolean union while capping pixel intensities at 255.
- **Detailed Explanation:** If mask 1 has value 255 and mask 2 has value 255 at an overlapping pixel, arithmetic addition (`+`) would overflow uint8 (wrapping $255+255 \to 254$). `np.maximum(m1, m2)` implements the mathematical set union $\mathbf{M} = \mathbf{M}_1 \cup \mathbf{M}_2$ cleanly.
- **Code Reference:** `common/dataset.py:56`
- **Formula:** $M_{\text{merged}}(x, y) = \max(M_1(x, y), M_2(x, y))$.

#### Q14: What is the native resolution of images in the BUSI dataset?
- **Direct Answer:** Native resolutions vary across scans, averaging around $500 \times 600$ pixels (ranging from $354 \times 415$ to $820 \times 748$).
- **Detailed Explanation:** Because clinical ultrasound probes have varying depths and aspect ratios, images are non-uniform. Case `benign (104)` is $777 \times 578$ pixels.
- **Code Reference:** `experiments/experiments.sqlite`

#### Q15: How does your system standardize image dimensions for deep learning?
- **Direct Answer:** All images and masks are resized to a standardized $256 \times 256 \times 1$ tensor for model training and inference.
- **Detailed Explanation:** Fixed dimensions are required for batch tensor processing in PyTorch. For clinical display on the receiver, predicted masks are upsampled back to native uncropped dimensions ($777 \times 578$).
- **Code Reference:** `models/unet.py`, `ui/receiver/segmentation_view.py`

#### Q16: How are normal cases (healthy tissue) represented in the mask files?
- **Direct Answer:** Normal cases have binary masks containing entirely zero pixels (all black).
- **Detailed Explanation:** Because there are no neoplasms in healthy scans, the ground-truth mask has zero foreground pixels ($|Y| = 0$).
- **Code Reference:** `common/dataset.py:65-72`

---

# CATEGORY 3: TRAIN/VAL/TEST PARTITIONING & DATA LEAKAGE PREVENTION

#### Q17: What exact dataset split ratio was used in this project?
- **Direct Answer:** 70% Training (546 images), 15% Validation (117 images), and 15% Test (117 images).
- **Detailed Explanation:** This guarantees a substantial, statistically significant held-out test set (117 images) to evaluate generalization without touching the training process.
- **Code Reference:** `data/BUSI/splits.json`
- **Formula:** $546 + 117 + 117 = 780$.

#### Q18: What is stratified sampling and why is it essential for medical datasets?
- **Direct Answer:** Stratified sampling ensures that every data split maintains the exact same proportion of classes (benign, malignant, normal) as the overall dataset.
- **Detailed Explanation:** Without stratification, a random split could place a disproportionate number of rare malignant cases into the test set, creating class imbalance and biased training.
- **Code Reference:** `common/dataset.py:train_test_split(..., stratify=labels)`

#### Q19: What is data leakage and how did your architecture prevent it?
- **Direct Answer:** Data leakage occurs when test data influences model training, producing overly optimistic, non-generalizable results.
- **Detailed Explanation:** We strictly prevented leakage by generating the split once with fixed seed 42, saving it to `data/BUSI/splits.json`. Test images were quarantined; only `splits['train']` was seen by model optimizers.
- **Code Reference:** `data/BUSI/splits.json`

#### Q20: Why is a fixed random seed (seed 42) used throughout the codebase?
- **Direct Answer:** To ensure strict scientific reproducibility across operating systems, runs, and evaluations.
- **Detailed Explanation:** Setting `random.seed(42)`, `np.random.seed(42)`, and `torch.manual_seed(42)` guarantees that splits, weight initializations, and augmentations are 100% deterministic.
- **Code Reference:** `common/dataset.py:32`, `training/train_mgda.py:45`

#### Q21: What is the exact class distribution in your 117-image held-out test set?
- **Direct Answer:** Exactly 66 Benign, 31 Malignant, and 20 Normal images.
- **Detailed Explanation:** Derived from 15% of each category: $437 \times 0.15 \approx 66$, $210 \times 0.15 \approx 31$, $133 \times 0.15 \approx 20$.
- **Code Reference:** `data/BUSI/splits.json`

#### Q22: What is patient-level vs. slice-level splitting in medical imaging?
- **Direct Answer:** Patient-level splitting ensures all images/slices from a single patient reside strictly in either train, val, or test—never split across them.
- **Detailed Explanation:** If two scans from the same patient appear in both train and test, the model can memorize patient-specific acoustic speckle rather than learning pathology, causing data leakage.
- **Code Reference:** `docs/limitations.md`

#### Q23: Why do we keep validation data separate from test data?
- **Direct Answer:** Validation data is used for tuning hyperparameters and early stopping; test data is evaluated only once to assess true real-world generalization.
- **Detailed Explanation:** Using test data for early stopping turns the test set into a validation set, introducing subtle selection bias.
- **Code Reference:** `training/train_mgda.py:150-180`

#### Q24: How does the system handle corrupt or unreadable image files during loading?
- **Direct Answer:** `common/dataset.py` checks `if img is None:` after `cv2.imread()`, logging a warning and safely skipping corrupted files without crashing.
- **Detailed Explanation:** Defensive input validation prevents `AttributeError: 'NoneType' object has no attribute 'shape'` during batch processing.
- **Code Reference:** `common/dataset.py:88-94`

---

# CATEGORY 4: DIGITAL IMAGE PROCESSING IN B-MODE ULTRASOUND

#### Q25: Why are ultrasound images represented as single-channel grayscale rather than 3-channel RGB?
- **Direct Answer:** B-mode ultrasound is fundamentally a 1D scalar acoustic backscatter intensity map; storing RGB channels triples byte size with zero diagnostic gain.
- **Detailed Explanation:** In RGB ultrasound files, $R = G = B$. Storing three identical channels wastes 66.7% of network bandwidth and GPU RAM.
- **Code Reference:** `common/dataset.py:cv2.IMREAD_GRAYSCALE`

#### Q26: What is the dynamic range and bit depth of standard ultrasound images?
- **Direct Answer:** 8-bit unsigned integer (`uint8`), providing 256 discrete gray levels from 0 (black) to 255 (white).
- **Detailed Explanation:** Intensity maps acoustic impedance: hypoechoic lesions are dark (0–50), normal parenchyma is mid-gray (60–160), and fibrous tissue/bone is hyperechoic white (180–255).
- **Code Reference:** `common/metrics.py:255`

#### Q27: Why must bilinear interpolation be used for images and nearest-neighbor for masks?
- **Direct Answer:** Bilinear interpolation creates fractional gray values ($0 < x < 255$) that corrupt binary masks, whereas nearest-neighbor strictly preserves discrete $\{0, 1\}$ or $\{0, 255\}$ labels.
- **Detailed Explanation:** For continuous grayscale images, bilinear interpolation (`cv2.INTER_LINEAR`) smooths transitions. For binary masks, nearest-neighbor (`cv2.INTER_NEAREST`) prevents threshold ambiguity.
- **Code Reference:** `common/dataset.py:102-108`

#### Q28: What is acoustic speckle noise, and why is it multiplicative?
- **Direct Answer:** Speckle is a granular interference pattern resulting from constructive and destructive phase interference of backscattered acoustic waves from sub-wavelength tissue scatterers.
- **Detailed Explanation:** It is modeled as $I(x, y) = S(x, y) \cdot \eta(x, y)$, where $\eta$ follows a Rayleigh distribution. It is signal-dependent, not additive white Gaussian noise.
- **Code Reference:** `docs/final_evaluation_member_1.md`
- **Formula:** $I = S \cdot \eta$.

#### Q29: What is acoustic shadowing, and how does it appear in ultrasound?
- **Direct Answer:** A dark, hypoechoic band extending deep to a highly attenuating structure (such as a malignant tumor or calcification) where sound waves cannot penetrate.
- **Detailed Explanation:** Posterior acoustic shadowing is a key diagnostic indicator of malignancy in breast ultrasound.
- **Code Reference:** `docs/research_summary_report.md`

#### Q30: How does contrast-to-noise ratio (CNR) impact lesion detectability?
- **Direct Answer:** Lower CNR makes distinguishing tumor boundaries from surrounding parenchymal tissue more difficult for both radiologists and AI algorithms.
- **Detailed Explanation:** $\text{CNR} = \frac{|\mu_{\text{lesion}} - \mu_{\text{background}}|}{\sigma_{\text{background}}}$. Dense acoustic speckle elevates $\sigma$, lowering CNR.
- **Formula:** $\text{CNR} = \frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}$.

#### Q31: How are pixel intensities normalized before feeding into neural networks?
- **Direct Answer:** Scaled from $[0, 255]$ integer range to $[0.0, 1.0]$ floating-point range via division by 255.0.
- **Detailed Explanation:** Normalization prevents numerical overflow, ensures zero-mean centered gradients, and stabilizes batch normalization layers.
- **Code Reference:** `receiver/segmentation_engine.py:45`
- **Formula:** $I_{\text{norm}} = \frac{I}{255.0}$.

#### Q32: What is the effect of aspect ratio distortion when resizing images?
- **Direct Answer:** Non-uniform resizing stretches or squashes circular lesions into artificial ovals, distorting morphological aspect ratio features used to differentiate benign from malignant masses.
- **Detailed Explanation:** Benign tumors are typically wider-than-tall; malignant tumors are taller-than-wide. Distorting aspect ratio impairs clinical diagnosis.
- **Code Reference:** `ui/receiver/segmentation_view.py:ResizableImageLabel`

---

# CATEGORY 5: MORPHOLOGICAL OPERATIONS & MARGIN EXPANSION

#### Q33: What is morphological dilation?
- **Direct Answer:** A mathematical morphology operation that expands the boundaries of foreground objects in a binary image using a structuring element.
- **Detailed Explanation:** Dilation sets an output pixel to 1 if the structuring element overlaps at least one foreground pixel of the input image.
- **Code Reference:** `sender/adaptive_compressor.py:118`
- **Formula:** $A \oplus B = \{ z \mid (\hat{B})_z \cap A \neq \emptyset \}$.

#### Q34: What structuring element shape was chosen in `sender/adaptive_compressor.py`?
- **Direct Answer:** An elliptical structuring element (`cv2.MORPH_ELLIPSE`).
- **Detailed Explanation:** Biological neoplasms expand radially into surrounding tissue. An elliptical kernel expands boundaries smoothly and organically, whereas rectangular kernels introduce artificial $90^\circ$ corners.
- **Code Reference:** `sender/adaptive_compressor.py:115`

#### Q35: How is kernel size calculated from the user-configured margin parameter $m$?
- **Direct Answer:** $\text{kernel\_size} = 2m + 1$.
- **Detailed Explanation:** An odd integer diameter ensures a well-defined center origin pixel for the structuring element. For $m=15\text{ px}$, $\text{kernel\_size} = 31\text{ px}$.
- **Code Reference:** `sender/adaptive_compressor.py:114`
- **Formula:** $K = 2m + 1$.

#### Q36: Why is a 15-pixel margin applied around the detected lesion?
- **Direct Answer:** To protect the peritumoral transition zone from compression artifacts and absorb any minor edge localization errors from the edge ROI detector.
- **Detailed Explanation:** The peritumoral boundary carries critical clinical indicators (spiculations, microlobulations). If compressed at Q20, these subtle edges would be destroyed.
- **Code Reference:** `sender/adaptive_compressor.py:default margin_px=15`

#### Q37: What is morphological erosion, and how does it relate to dilation?
- **Direct Answer:** Erosion shrinks object boundaries by keeping an output pixel as 1 only if the structuring element completely fits inside the foreground object.
- **Detailed Explanation:** Dilation and erosion are dual operations: $(A \ominus B)^c = A^c \oplus \hat{B}$.
- **Formula:** $A \ominus B = \{ z \mid (B)_z \subseteq A \}$.

#### Q38: What is morphological opening and closing?
- **Direct Answer:** Opening is erosion followed by dilation (removes small isolated noise spots). Closing is dilation followed by erosion (fills small internal holes).
- **Detailed Explanation:** $\text{Open}(A, B) = (A \ominus B) \oplus B$; $\text{Close}(A, B) = (A \oplus B) \ominus B$.

#### Q39: Can margin expansion cause the ROI mask to exceed image boundaries?
- **Direct Answer:** No, OpenCV's `cv2.dilate` automatically clips boundary coordinates to image array dimensions $(H, W)$.
- **Detailed Explanation:** Matrix slicing prevents index-out-of-bounds errors when expanding lesions located near the edge of the scan.
- **Code Reference:** `sender/adaptive_compressor.py`

#### Q40: What happens if an ultrasound scan contains multiple distinct lesions during dilation?
- **Direct Answer:** Every distinct lesion is dilated simultaneously by the structuring element, organically merging any adjacent foci that are closer than $2m$ pixels apart.
- **Detailed Explanation:** This unified multi-focal expansion ensures comprehensive protection of all tumor foci in complex scans.
- **Code Reference:** `sender/adaptive_compressor.py:120`

---

# CATEGORY 6: DISCRETE COSINE TRANSFORM (DCT) MATHEMATICS

#### Q41: What is the 2D Discrete Cosine Transform (DCT)?
- **Direct Answer:** A mathematical transform that maps an $8 \times 8$ spatial pixel matrix into 64 spatial frequency coefficients.
- **Detailed Explanation:** DCT expresses spatial signals as sums of sinusoids of varying frequencies and amplitudes, concentrating image energy into low-frequency coefficients.
- **Code Reference:** `common/compression.py`

#### Q42: Write down the mathematical formula for the 2D-DCT-II.
- **Direct Answer:**
  $$F(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left[ \frac{(2x+1)u\pi}{16} \right] \cos\left[ \frac{(2y+1)v\pi}{16} \right]$$
  where $C(w) = \frac{1}{\sqrt{2}}$ for $w=0$, and $C(w) = 1$ for $w > 0$.

#### Q43: What is the DC coefficient, and what does it represent?
- **Direct Answer:** The coefficient $F(0, 0)$, representing zero spatial frequency—it equals 8 times the average intensity of the $8 \times 8$ pixel block.
- **Detailed Explanation:** Because $u=0, v=0$, the cosine terms evaluate to 1, leaving $F(0, 0) = \frac{1}{8} \sum \sum f(x, y) = 8 \times \mu_{\text{block}}$.
- **Formula:** $F(0, 0) = 8 \cdot \mu_{\text{block}}$.

#### Q44: What are AC coefficients?
- **Direct Answer:** The remaining 63 coefficients $F(u, v)$ for $(u, v) \neq (0, 0)$, representing increasingly higher horizontal, vertical, and diagonal spatial frequencies.
- **Detailed Explanation:** Higher indices correspond to rapid pixel intensity fluctuations across the $8 \times 8$ block.

#### Q45: Why is DCT preferred over Discrete Fourier Transform (DFT) in image compression?
- **Direct Answer:** DCT has superior energy compaction efficiency and produces purely real-valued coefficients, avoiding complex numbers and boundary discontinuities.
- **Detailed Explanation:** DFT assumes periodic signal extension, creating sharp boundary discontinuities that generate high-frequency artifacts. DCT assumes even symmetric extension, avoiding boundary leaps.

#### Q46: What is the inverse 2D-DCT formula?
- **Direct Answer:**
  $$f(x, y) = \frac{1}{4} \sum_{u=0}^{7} \sum_{v=0}^{7} C(u) C(v) F(u, v) \cos\left[ \frac{(2x+1)u\pi}{16} \right] \cos\left[ \frac{(2y+1)v\pi}{16} \right]$$

#### Q47: Why is an $8 \times 8$ block size used rather than transforming the entire image at once?
- **Direct Answer:** $8 \times 8$ blocks balance computational efficiency with local spatial correlation, whereas a full-image DCT is computationally expensive ($O(N^2 \log N)$) and loses local spatial localization.
- **Detailed Explanation:** Natural image correlation drops sharply beyond 8 pixels. Transforming small blocks adapts to local texture variations.

#### Q48: Why are pixel values level-shifted by -128 before computing the forward DCT?
- **Direct Answer:** To convert unsigned 8-bit integers $[0, 255]$ into signed values $[-128, 127]$ centered around zero, reducing dynamic range requirements for the DC coefficient.
- **Detailed Explanation:** Zero-centering prevents numerical bias in arithmetic computations.

---

# CATEGORY 7: STANDARD JPEG COMPRESSION PIPELINE & BLOCK PARTITIONING

#### Q49: List the 5 sequential stages of standard JPEG compression.
- **Direct Answer:**
  1. $8 \times 8$ Block Partitioning and Level Shifting (-128).
  2. Forward 2D Discrete Cosine Transform (2D-DCT).
  3. Scalar Quantization (the only lossy step).
  4. Zigzag Reordering.
  5. Run-Length Encoding (RLE) and Huffman Entropy Coding.

#### Q50: How does JPEG handle images whose dimensions are not multiples of 8?
- **Direct Answer:** By padding the right and bottom boundaries with edge pixel replicates to complete the final $8 \times 8$ blocks, trimming the padding after decoding.
- **Detailed Explanation:** Ensures standard $8 \times 8$ block matrix operations can execute uniformly across the entire image.

#### Q51: Which library backend performs JPEG compression in your project?
- **Direct Answer:** OpenCV's `cv2.imencode('.jpg', ...)` backed by libjpeg-turbo.
- **Detailed Explanation:** libjpeg-turbo uses SIMD assembly instructions to achieve ultra-fast DCT and Huffman processing.
- **Code Reference:** `common/compression.py:28`

#### Q52: How is JPEG quality factor passed to OpenCV in Python?
- **Direct Answer:** Via the parameter list `[int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]`.
- **Detailed Explanation:** Accepts integer values from 1 (maximum compression, lowest quality) to 100 (minimal compression, highest quality).
- **Code Reference:** `common/compression.py:29`

#### Q53: What are the four discrete JPEG quality levels evaluated in your project?
- **Direct Answer:** Q90, Q70, Q50, and Q30.
- **Detailed Explanation:** These represent clinical regimes: Q90 is near-lossless, Q70 is balanced, Q50 shows visible compression, and Q30 represents severe degradation.
- **Code Reference:** `common/compression.py`

#### Q54: What is the typical compression ratio achieved by standard uniform Q90 on BUSI images?
- **Direct Answer:** Between $5:1$ and $8:1$ (approx. 82% to 85% byte size reduction).
- **Detailed Explanation:** On case `benign (104)`, raw 455,471 bytes compressed to 78,050 bytes ($\text{CR} = 5.84:1$).
- **Code Reference:** `experiments/experiments.sqlite`

#### Q55: What is the typical compression ratio achieved by uniform Q30?
- **Direct Answer:** Between $18:1$ and $25:1$ (approx. 94% to 96% byte size reduction).
- **Detailed Explanation:** On case `benign (104)`, raw 455,471 bytes compressed to 24,528 bytes ($\text{CR} = 18.57:1$).
- **Code Reference:** `experiments/experiments.sqlite`

#### Q56: Why does JPEG decompression occur entirely in memory in your receiver?
- **Direct Answer:** To maximize throughput and avoid disk I/O bottlenecks by decoding byte streams directly from memory buffers using `cv2.imdecode`.
- **Detailed Explanation:** Writing temporary `.jpg` files to SSD/HDD for every received frame introduces 5–15 ms of disk latency.
- **Code Reference:** `receiver/tcp_server.py:145`

---

# CATEGORY 8: SCALAR QUANTIZATION & LOSS MECHANICS

#### Q57: What is the mathematical definition of JPEG scalar quantization?
- **Direct Answer:**
  $$F_Q(u, v) = \text{round}\left( \frac{F(u, v)}{Q(u, v)} \right)$$
- **Detailed Explanation:** Each DCT coefficient $F(u, v)$ is divided by its corresponding entry in the quantization matrix $Q(u, v)$ and rounded to the nearest integer.

#### Q58: Why is quantization the only lossy step in JPEG compression?
- **Direct Answer:** Because integer rounding permanently truncates fractional remainders, discarding exact precision; DCT, zigzag scanning, and Huffman coding are 100% mathematically reversible.
- **Detailed Explanation:** Dequantization multiplies by $Q(u, v)$ ($F'(u, v) = F_Q(u, v) \cdot Q(u, v)$), but cannot recover the discarded fractional values.

#### Q59: How is the base luminance quantization matrix $Q_{50}$ scaled for quality factor $Q$?
- **Direct Answer:** Scaled by factor $S$:
  $$S = \begin{cases} \frac{5000}{Q} & \text{if } Q < 50 \\ 200 - 2Q & \text{if } Q \ge 50 \end{cases}$$
  $$Q_{\text{scaled}}(u, v) = \text{clip}\left( \text{round}\left( \frac{Q_{50}(u, v) \cdot S + 50}{100} \right), 1, 255 \right)$$

#### Q60: What happens to quantization matrix entries when Quality = 100?
- **Direct Answer:** The scaling factor $S = 200 - 2(100) = 0$, clamping all quantization table entries to 1 ($Q(u, v) = 1$).
- **Detailed Explanation:** Coefficients are divided by 1, minimizing rounding errors, though minor DCT floating-point roundoff persists.

#### Q61: What happens to quantization matrix entries when Quality = 30?
- **Direct Answer:** The scaling factor $S = 5000 / 30 \approx 166.67$, multiplying quantization divisors by $\approx 1.67\times$.
- **Detailed Explanation:** Large divisors force high-frequency AC coefficients to round to zero, causing heavy loss of fine detail.

#### Q62: Why are quantization table values larger in the bottom-right corner than the top-left?
- **Direct Answer:** Because the human visual system is much less sensitive to high spatial frequencies (bottom-right) than low spatial frequencies (top-left).
- **Detailed Explanation:** Psycho-visual models allow high frequencies to be coarsely quantized without perceived degradation in natural scenes.

#### Q63: Why does high quantization impair ultrasound imaging more than natural photos?
- **Direct Answer:** Because ultrasound speckle noise and subtle acoustic margins reside in high spatial frequencies; coarse quantization treats them as discardable noise, destroying diagnostic texture.
- **Detailed Explanation:** Natural photos have smooth gradients; medical ultrasound relies on fine textural backscatter.

#### Q64: What is dequantization on the receiver?
- **Direct Answer:** The inverse operation multiplying quantized coefficients by the identical quantization table: $F'(u, v) = F_Q(u, v) \cdot Q(u, v)$.
- **Detailed Explanation:** Reconstructs approximate DCT coefficients prior to computing the inverse 2D-DCT.

---

# CATEGORY 9: ZIGZAG SCANNING & HUFFMAN ENTROPY CODING

#### Q65: What is Zigzag scanning in JPEG?
- **Direct Answer:** A diagonal traversal path through the $8 \times 8$ matrix that orders coefficients from lowest spatial frequency (top-left) to highest spatial frequency (bottom-right).
- **Detailed Explanation:** Traverses $(0,0) \to (0,1) \to (1,0) \to (2,0) \to (1,1) \dots$, grouping non-zero coefficients first followed by long runs of zeros.

#### Q66: Why is zigzag scanning essential for compression?
- **Direct Answer:** It clusters zeroed high-frequency AC coefficients at the end of the 1D sequence, enabling massive compression via Run-Length Encoding and End-of-Block markers.
- **Detailed Explanation:** Transforming a 2D matrix into a 1D sequence optimized for run-length encoding maximizes entropy coding efficiency.

#### Q67: What is an End-of-Block (EOB) marker in JPEG?
- **Direct Answer:** A special Huffman code indicating that all remaining AC coefficients in the $8 \times 8$ block are zero.
- **Detailed Explanation:** Eliminates the need to encode trailing zeros individually, achieving massive bit savings in smooth blocks.

#### Q68: What is Run-Length Encoding (RLE) in JPEG?
- **Direct Answer:** An encoding scheme that represents consecutive zeros as a pair: `(number of preceding zeros, amplitude category of next non-zero coefficient)`.
- **Detailed Explanation:** Compresses sparse coefficient arrays efficiently into compact symbol pairs.

#### Q69: What is Huffman Coding?
- **Direct Answer:** A lossless entropy coding algorithm that assigns variable-length binary prefix codes to symbols based on their statistical frequency: frequent symbols receive short codes, rare symbols receive long codes.
- **Detailed Explanation:** Huffman coding approaches the theoretical Shannon entropy limit: $H(X) = -\sum P(x) \log_2 P(x)$.

#### Q70: Is Huffman coding lossy or lossless?
- **Direct Answer:** 100% mathematically lossless.
- **Detailed Explanation:** The prefix tree guarantees unique, exact decodability; every original symbol is reconstructed bit-for-bit.

#### Q71: What is the difference between standard and custom Huffman tables?
- **Direct Answer:** Standard tables use pre-computed empirical frequencies from typical images; custom (optimized) tables compute exact frequency distributions for each specific image.
- **Detailed Explanation:** Custom tables yield 2–5% better compression but require encoding the table in the JPEG header.

#### Q72: What happens if a single bit flips in a Huffman-coded bitstream?
- **Direct Answer:** Catastrophic desynchronization: because Huffman codes have variable lengths, a bit flip causes the decoder to read incorrect code lengths, corrupting all subsequent pixels until a restart marker.
- **Detailed Explanation:** This is why TCP's zero-loss guarantee and our SHA-256 integrity verification are mandatory for medical transmission.

---

# CATEGORY 10: COMPRESSION ARTIFACTS IN MEDICAL ULTRASOUND

#### Q73: What are $8 \times 8$ blocking artifacts?
- **Direct Answer:** Visible artificial grid lines that appear along the boundaries of adjacent $8 \times 8$ pixel blocks under heavy compression.
- **Detailed Explanation:** Caused by independent block-by-block quantization; coarse quantization forces adjacent blocks to have mismatched edge intensities, creating artificial linear boundaries.

#### Q74: What is Gibbs ringing artifact?
- **Direct Answer:** An oscillatory ripple or halo pattern appearing adjacent to sharp, high-contrast boundaries in an image.
- **Detailed Explanation:** Caused by truncating high-frequency Fourier/DCT coefficients; approximating a step edge with truncated sinusoids causes overshoot and ringing.

#### Q75: How do blocking artifacts affect convolutional neural networks (CNNs)?
- **Direct Answer:** CNN edge-detection kernels ($3 \times 3$ Sobel-like filters) falsely activate on the artificial rectilinear $8 \times 8$ grid lines, corrupting feature representations and producing fragmented segmentation masks.
- **Detailed Explanation:** Artificial edges compete with real anatomical edges during convolutional feature extraction.

#### Q76: What is blurring artifact in JPEG?
- **Direct Answer:** The loss of sharpness and fine texture caused by zeroing out high-frequency AC coefficients during coarse quantization.
- **Detailed Explanation:** In ultrasound, blurring erases fine acoustic speckle, smoothing out micro-calcifications and subtle acoustic margins.

#### Q77: At which quality factor do blocking artifacts become prominent in your tests?
- **Direct Answer:** At Q50 they become noticeable, and at Q30 they become severe across the entire image.
- **Detailed Explanation:** At Q30, high-frequency AC coefficients are almost entirely zeroed out, reducing blocks to flat DC planes.
- **Code Reference:** `docs/final_evaluation_results_and_limitations.md`

#### Q78: Why does Region-Adaptive compression prevent blocking artifacts inside the lesion?
- **Direct Answer:** Because the lesion and its margin are encoded at pristine Q90 quality, where quantization divisors are minimal and high-frequency coefficients are fully preserved.
- **Detailed Explanation:** Blocking artifacts are confined to the non-diagnostic background at Q20, leaving the tumor boundary pristine.
- **Code Reference:** `sender/adaptive_compressor.py`

#### Q79: What is color bleeding / chroma subsampling artifact, and does it affect this project?
- **Direct Answer:** No, chroma subsampling artifacts (e.g., 4:2:0 subsampling) only affect color images; our ultrasound pipeline processes pure single-channel 8-bit grayscale images.

#### Q80: How does Gaussian feathering eliminate boundary transition artifacts between ROI and background?
- **Direct Answer:** By replacing sharp binary cutoffs with a smooth, continuous alpha gradient ($\sigma = m/3$), preventing artificial step edges that would trigger Gibbs ringing.
- **Detailed Explanation:** Continuous intensity blending keeps spatial frequency energy smooth, avoiding spurious edge activations in CNNs.
- **Code Reference:** `sender/adaptive_compressor.py:124-128`

---

# CATEGORY 11: OBJECTIVE IMAGE QUALITY METRICS: MSE & PSNR

#### Q81: What is Mean Squared Error (MSE)? Give the formula.
- **Direct Answer:** The average squared difference between corresponding pixels of original image $I$ and compressed image $K$:
  $$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I(x, y) - K(x, y)]^2$$
- **Code Reference:** `common/metrics.py:compute_mse`

#### Q82: What is Peak Signal-to-Noise Ratio (PSNR)? Give the formula.
- **Direct Answer:**
  $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}} \right)$$
  for 8-bit images with maximum intensity $MAX_I = 255$.
- **Code Reference:** `common/metrics.py:compute_psnr`

#### Q83: Why is PSNR measured on a logarithmic decibel scale?
- **Direct Answer:** Because human perceptual sensitivity to sensory stimuli is logarithmic (Weber-Fechner Law), and pixel error spans multiple orders of magnitude.

#### Q84: What does a PSNR value of 100 dB represent in your code?
- **Direct Answer:** It represents $\text{MSE} = 0$ (perfect identity); capped at 100.0 dB to prevent floating-point division-by-zero crashes.
- **Code Reference:** `common/metrics.py:42`

#### Q85: What are standard clinical benchmark ranges for PSNR?
- **Direct Answer:** $>40\text{ dB}$: Excellent (visually lossless). $35\text{–}40\text{ dB}$: Good. $30\text{–}35\text{ dB}$: Marginal. $<30\text{ dB}$: Poor/unacceptable.

#### Q86: What is ROI-PSNR and how does it differ from Full-Image PSNR?
- **Direct Answer:** ROI-PSNR computes MSE exclusively across pixels within the dilated lesion boundary, isolating clinical diagnostic fidelity from background compression.
- **Detailed Explanation:** Full-image PSNR is dominated by background pixels; ROI-PSNR directly measures distortion where it matters clinically.
- **Code Reference:** `common/metrics.py:compute_roi_psnr`
- **Formula:** $\text{MSE}_{\text{ROI}} = \frac{1}{|\text{ROI}|} \sum_{(x,y) \in \text{ROI}} [I(x,y) - K(x,y)]^2$.

#### Q87: In trial `EXP_ADAPT_1795063124`, what was the Full PSNR vs ROI PSNR for `benign (104)`?
- **Direct Answer:** Full-Image PSNR was **$36.42\text{ dB}$**, while Preserved ROI PSNR was **$49.70\text{ dB}$**.
- **Detailed Explanation:** Demonstrates that while aggressive background compression lowers global PSNR, the lesion itself remains preserved at near-lossless 49.70 dB fidelity.
- **Code Reference:** `experiments/experiments.sqlite`

#### Q88: Why can an image with lower global PSNR achieve a higher clinical segmentation score?
- **Direct Answer:** Because global PSNR averages error across all pixels uniformly; an image with heavy background compression but a pristine lesion will have lower global PSNR but superior segmentation accuracy compared to an image with uniform blur across the tumor.

---

# CATEGORY 12: STRUCTURAL SIMILARITY INDEX (SSIM)

#### Q89: What is SSIM and why was it developed?
- **Direct Answer:** Structural Similarity Index; developed by Wang et al. (2004) to measure perceived degradation by evaluating luminance, contrast, and structural correlation rather than raw pixel differences.
- **Detailed Explanation:** Models human visual system characteristics; two images with identical MSE can have vastly different structural fidelity.

#### Q90: State the mathematical formula for SSIM.
- **Direct Answer:**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- **Code Reference:** `common/metrics.py:compute_ssim`

#### Q91: What are $C_1$ and $C_2$ in the SSIM equation?
- **Direct Answer:** Small stability constants: $C_1 = (K_1 L)^2 = (0.01 \times 255)^2 = 6.5025$ and $C_2 = (K_2 L)^2 = (0.03 \times 255)^2 = 58.5225$.
- **Detailed Explanation:** Prevent division-by-zero crashes when local means or variances approach zero in black regions.

#### Q92: What window size is used to compute local SSIM statistics?
- **Direct Answer:** An $11 \times 11$ circular Gaussian sliding window with standard deviation $\sigma = 1.5$.
- **Detailed Explanation:** Slides across the image to generate a local SSIM quality map, averaged to yield Mean SSIM (MSSIM).

#### Q93: What is the dynamic range of SSIM?
- **Direct Answer:** Bounded between $-1.0$ and $+1.0$; for practical non-negative image intensities, it spans $0.0$ (no correlation) to $1.0$ (perfect identity).

#### Q94: What was the SSIM of `benign (104)` under Region-Adaptive compression?
- **Direct Answer:** Exactly **$0.9292$**.
- **Detailed Explanation:** Reflects high structural fidelity overall, with slight reduction due to aggressive Q20 background compression.
- **Code Reference:** `experiments/experiments.sqlite`

#### Q95: What library function is used to calculate SSIM in your codebase?
- **Direct Answer:** `skimage.metrics.structural_similarity(orig, comp, data_range=255)`.
- **Code Reference:** `common/metrics.py:58`

#### Q96: What are the three components that multiply to form SSIM?
- **Direct Answer:** Luminance comparison $l(x, y)$, contrast comparison $c(x, y)$, and structural comparison $s(x, y)$.
- **Formula:** $\text{SSIM} = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma$ (with $\alpha=\beta=\gamma=1$).

---

# CATEGORY 13: COMPRESSION RATIO, BITRATE & SIZE REDUCTION

#### Q97: State the formula for Compression Ratio (CR).
- **Direct Answer:**
  $$\text{CR} = \frac{\text{Original File Size in Bytes}}{\text{Compressed File Size in Bytes}} = \frac{S_{\text{orig}}}{S_{\text{comp}}}$$
- **Code Reference:** `common/compression.py:compute_compression_stats`

#### Q98: State the formula for Percentage Size Reduction.
- **Direct Answer:**
  $$\text{Size Reduction } \% = \left( 1 - \frac{S_{\text{comp}}}{S_{\text{orig}}} \right) \times 100\%$$
- **Code Reference:** `common/compression.py:52`

#### Q99: State the formula for Bitrate in bits-per-pixel (bpp).
- **Direct Answer:**
  $$\text{bpp} = \frac{S_{\text{comp}} \times 8}{M \times N} = \frac{8}{\text{CR}}$$
  for 8-bit grayscale images.

#### Q100: If an image has CR = 8:1, what is its bitrate and percentage size reduction?
- **Direct Answer:** $\text{Bitrate} = 8 / 8 = 1.0\text{ bpp}$; $\text{Size Reduction} = (1 - 1/8) \times 100\% = 87.5\%$.

#### Q101: What was the byte size of `benign (104)` raw uncompressed vs. Region-Adaptive?
- **Direct Answer:** Raw uncompressed: **$455,471\text{ bytes}$** ($444.8\text{ KB}$); Region-Adaptive: **$48,481\text{ bytes}$** ($47.34\text{ KB}$).
- **Detailed Explanation:** Yielding a Compression Ratio of **$9.39:1$** and size reduction of **$89.36\%$**.
- **Code Reference:** `experiments/experiments.sqlite`

#### Q102: How much byte savings did Region-Adaptive achieve over uniform Q90 on `benign (104)`?
- **Direct Answer:** **$29,569\text{ bytes}$ saved**, representing a **$37.9\%$ byte reduction** ($78,050\text{ B} \to 48,481\text{ B}$).
- **Formula:** $\frac{78,050 - 48,481}{78,050} \times 100\% = 37.88\%$.

#### Q103: Does compression ratio depend on image content?
- **Direct Answer:** Yes, images with smooth, uniform regions achieve much higher compression ratios than images with dense, complex acoustic speckle noise.

#### Q104: Why is bpp a better metric than file size when comparing across different image resolutions?
- **Direct Answer:** Because bpp normalizes byte payload against spatial resolution ($M \times N$), enabling fair comparison between images of varying dimensions.

---

# CATEGORY 14: BIOMEDICAL U-NET ARCHITECTURE & SKIP CONNECTIONS

#### Q105: What is the total parameter count and weight file size of your clinical U-Net?
- **Direct Answer:** Exactly **$7,762,465$ parameters**, occupying **$31.2\text{ MB}$** on disk (`models/unet_busi.pth`).
- **Code Reference:** `models/unet.py`

#### Q106: Describe the 4 resolution stages of the U-Net encoder.
- **Direct Answer:** Stage 1: 64 channels; Stage 2: 128 channels; Stage 3: 256 channels; Stage 4: 512 channels. Spatial resolution is halved at each stage via $2 \times 2$ max pooling ($256 \to 128 \to 64 \to 32 \to 16$).
- **Code Reference:** `models/unet.py:UNet`

#### Q107: What is the bottleneck layer in your U-Net?
- **Direct Answer:** Two $3 \times 3$ convolutional layers with **1024 channels** operating at $16 \times 16$ spatial resolution, capturing abstract global context.

#### Q108: What are skip connections and why are they vital in U-Net?
- **Direct Answer:** Lateral bridges that concatenate high-resolution feature maps from encoder stages directly into corresponding decoder stages.
- **Detailed Explanation:** Max pooling loses exact spatial coordinate information; skip connections bypass the bottleneck to restore fine-grained spatial boundaries.

#### Q109: What upsampling method is used in the U-Net decoder?
- **Direct Answer:** $2 \times 2$ Transposed Convolution (`nn.ConvTranspose2d`) with stride 2 and padding 0.
- **Detailed Explanation:** Contains learnable parameters that optimize feature reconstruction during backpropagation.

#### Q110: What activation functions are used in U-Net?
- **Direct Answer:** Rectified Linear Unit (`ReLU`, $f(x) = \max(0, x)$) after every convolution, and Sigmoid ($\sigma(z) = \frac{1}{1 + e^{-z}}$) on the final $1 \times 1$ layer.

#### Q111: What threshold $\tau$ converts sigmoid output into a binary segmentation mask?
- **Direct Answer:** Standard clinical threshold of $\tau = 0.5$ ($\hat{y} = 1$ if $\sigma(z) \ge 0.5$, else $0$).
- **Code Reference:** `receiver/segmentation_engine.py:predict`

#### Q112: What is the inference latency of the clinical U-Net on a standard CPU?
- **Direct Answer:** Average **$79.39\text{ ms}$** per image.
- **Code Reference:** `experiments/experiments.sqlite`

---

# CATEGORY 15: SEGMENTATION LOSS FUNCTIONS: BCE & DICE LOSS

#### Q113: State the mathematical formula for Binary Cross-Entropy (BCE) loss.
- **Direct Answer:**
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$$

#### Q114: State the mathematical formula for soft Dice loss.
- **Direct Answer:**
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum y_i \hat{p}_i + \epsilon}{\sum y_i + \sum \hat{p}_i + \epsilon}$$
  where $\epsilon = 10^{-6}$ is a smoothing constant.

#### Q115: Why is BCE loss alone inadequate for medical ultrasound segmentation?
- **Direct Answer:** Severe class imbalance: lesions occupy only 2–5% of pixels. Predicting all zeros achieves 95–98% BCE accuracy while having zero clinical diagnostic utility.

#### Q116: Why is Dice loss alone inadequate?
- **Direct Answer:** Dice loss has a non-convex optimization surface with vanishing gradients early in training when predictions $\hat{p}_i \approx 0$, causing training instability.

#### Q117: What is the combined loss formulation?
- **Direct Answer:** $\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$.

#### Q118: What is gradient conflict between BCE and Dice loss?
- **Direct Answer:** When their gradient vectors point in opposing directions ($\langle \nabla \mathcal{L}_{\text{BCE}}, \nabla \mathcal{L}_{\text{Dice}} \rangle < 0$), causing optimization steps to cancel each other out.

#### Q119: What is Focal Loss and how does it relate to BCE?
- **Direct Answer:** Focal Loss adds a modulating factor $(1 - \hat{p}_t)^\gamma$ to BCE to down-weight easy background examples and focus training on hard boundary pixels.

#### Q120: How does soft Dice loss differ from hard Dice score?
- **Direct Answer:** Soft Dice uses continuous probabilities $\hat{p} \in [0, 1]$ to allow gradient backpropagation; hard Dice applies a threshold $\tau = 0.5$ yielding discrete binary sets $\{0, 1\}$.

---

# CATEGORY 16: MULTI-GRADIENT DESCENT ALGORITHM (MGDA) & PARETO OPTIMIZATION

#### Q121: What is MGDA?
- **Direct Answer:** Multi-Gradient Descent Algorithm; a multi-objective optimization method that finds a common descent vector satisfying Karush-Kuhn-Tucker (KKT) Pareto-stationarity conditions.
- **Code Reference:** `training/mgda_solver.py:MGDASolver`

#### Q122: State the closed-form analytical formula for $\alpha^*$ in 2-objective MGDA.
- **Direct Answer:**
  $$\alpha^* = \text{clip}\left( \frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1 \right)$$
- **Code Reference:** `training/mgda_solver.py:find_min_norm_element`

#### Q123: What is the Frank-Wolfe algorithm's role in MGDA?
- **Direct Answer:** It solves the quadratic subproblem of finding the minimum-norm point within the convex hull of objective gradients.

#### Q124: What validation Dice score did MGDA achieve compared to baseline training?
- **Direct Answer:** MGDA achieved **$0.6581$** (Epoch 19) vs. **$0.6056$** (Epoch 21) for baseline scalar weighting—a **$+5.25\%$ absolute gain**.
- **Code Reference:** `docs/final_evaluation_results_and_limitations.md`

#### Q125: At which network layer were gradients extracted for MGDA?
- **Direct Answer:** At the shared representation bottleneck layer ($Z \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$).
- **Detailed Explanation:** Proven by Sener & Koltun to provide an upper bound on multi-objective loss while avoiding duplicate whole-network backward passes.

#### Q126: What does Pareto optimality mean in deep learning?
- **Direct Answer:** A state where no network parameter update can improve one loss objective without worsening at least one other loss objective.

#### Q127: What happens in MGDA if $\mathbf{g}_1$ and $\mathbf{g}_2$ point in identical directions?
- **Direct Answer:** $\mathbf{g}_1 = \mathbf{g}_2$, the denominator $\|\mathbf{g}_1 - \mathbf{g}_2\|^2 \to 0$, and the solver assigns equal weights $\alpha^* = 0.5, \beta^* = 0.5$.

#### Q128: What is the computational overhead of your MGDA solver during training?
- **Direct Answer:** Less than 18% additional training time per epoch, requiring zero additional GPU RAM by operating on bottleneck feature tensors.

---

# CATEGORY 17: LIGHTWEIGHT EDGE ROI DETECTOR

#### Q129: What is the parameter count and disk footprint of `ROIDetector`?
- **Direct Answer:** Exactly **$482,449$ parameters**, occupying **$1.96\text{ MB}$** on disk (`models/roi_detector_v1.pth`).
- **Code Reference:** `models/roi_detector.py`

#### Q130: How much smaller is `ROIDetector` compared to the clinical `UNet`?
- **Direct Answer:** **$16.1\times$ fewer parameters** ($482\text{K}$ vs $7.76\text{M}$) and **$15.9\times$ smaller disk size** ($1.96\text{ MB}$ vs $31.2\text{ MB}$).

#### Q131: What is the CPU inference latency of `ROIDetector`?
- **Direct Answer:** **$3.82\text{ ms}$** per image—over **$20.8\times$ faster** than the clinical U-Net ($79.39\text{ ms}$).

#### Q132: What was the held-out test performance of `ROIDetector`?
- **Direct Answer:** Test Dice = **$0.6087$**, Test IoU = **$0.5126$** across 117 quarantined held-out test scans.

#### Q133: What are the channel depths in `ROIDetector`?
- **Direct Answer:** $[16, 32, 64, 128]$ with a 256-channel bottleneck (exactly one-quarter the channel depth of clinical U-Net).
- **Code Reference:** `models/roi_detector.py:16`

#### Q134: Where does `ROIDetector` execute in the deployment topology?
- **Direct Answer:** Strictly on the **Sender Node (Ubuntu edge laptop)** to detect lesion coordinates prior to compression and transmission.

#### Q135: What happens if `ROIDetector` fails to detect a lesion ($|\hat{M}| = 0$)?
- **Direct Answer:** Automatic safety fallback: the system defaults to uniform Q70 compression to prevent lesion loss.
- **Code Reference:** `sender/adaptive_compressor.py:150`

#### Q136: Can `ROIDetector` run on systems without CUDA GPU accelerators?
- **Direct Answer:** Yes, with only 482K parameters and 3.8 ms CPU execution time, it runs entirely on commodity CPU threads.

---

# CATEGORY 18: REGION-ADAPTIVE COMPRESSION & GAUSSIAN FEATHERING

#### Q137: Describe the 4 mathematical steps of Region-Adaptive compression.
- **Direct Answer:**
  1. Detect lesion mask $M$ via `ROIDetector`.
  2. Dilate mask with elliptical kernel ($m=15\text{ px}$): $M_{\text{dilated}} = M \oplus B$.
  3. Apply Gaussian blur ($\sigma = m/3 = 5.0$) to create alpha blend mask $A(x, y) \in [0, 1]$.
  4. Blend dual-rate frames: $I_{\text{adaptive}} = A \cdot I_{\text{Q90}} + (1 - A) \cdot I_{\text{Q20}}$, then encode to standard JPEG.
- **Code Reference:** `sender/adaptive_compressor.py:compress_adaptive`

#### Q138: Why is Gaussian feathering ($\sigma = m/3$) used?
- **Direct Answer:** To replace sharp binary boundaries with a smooth spatial gradient, preventing artificial high-frequency edges that cause DCT Gibbs ringing and false CNN activations.

#### Q139: What are the default ROI and background quality factors?
- **Direct Answer:** $Q_{\text{ROI}} = 90$ and $Q_{\text{BG}} = 20$.

#### Q140: What happens if background quality is set below Q20 (e.g., Q10)?
- **Direct Answer:** Severe blocking artifacts bleed across the Gaussian blend zone, and secondary anatomical reference landmarks (like Cooper's ligaments) are erased.

#### Q141: What byte savings did Region-Adaptive compression achieve on `benign (104)`?
- **Direct Answer:** **$37.9\%$ byte reduction** ($78,050\text{ B} \to 48,481\text{ B}$) compared to uniform Q90.
- **Code Reference:** `experiments/experiments.sqlite`

#### Q142: Did Region-Adaptive compression degrade segmentation accuracy on `benign (104)`?
- **Direct Answer:** Zero degradation: Dice score was **$0.8664$** under both uniform Q90 and Region-Adaptive compression.

#### Q143: Does the receiver need the alpha mask to decode the image?
- **Direct Answer:** No, the transmitted payload is a standard, fully compliant JPEG image decodable by any standard JPEG viewer.

#### Q144: How is the alpha mask clamped to $[0.0, 1.0]$?
- **Direct Answer:** `np.clip(alpha, 0.0, 1.0).astype(np.float32)`.
- **Code Reference:** `sender/adaptive_compressor.py:126`

---

# CATEGORY 19: SEGMENTATION METRICS: DICE & IOU MATHEMATICS

#### Q145: State the formula for Dice Similarity Coefficient (DSC).
- **Direct Answer:**
  $$\text{Dice} = \frac{2 \cdot |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} = \frac{2 \cdot \text{TP}}{2\cdot\text{TP} + \text{FP} + \text{FN}}$$
- **Code Reference:** `common/metrics.py:compute_dice_score`

#### Q146: State the formula for Intersection over Union (IoU / Jaccard Index).
- **Direct Answer:**
  $$\text{IoU} = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
- **Code Reference:** `common/metrics.py:compute_iou_score`

#### Q147: State the conversion formulas between Dice and IoU.
- **Direct Answer:**
  $$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

#### Q148: Which metric is always lower: Dice or IoU?
- **Direct Answer:** IoU is always strictly less than or equal to Dice ($\text{IoU} \le \text{Dice}$).

#### Q149: If an AI achieves $\text{IoU} = 0.7643$, what is its Dice score?
- **Direct Answer:**
  $$\text{Dice} = \frac{2 \times 0.7643}{1 + 0.7643} = \frac{1.5286}{1.7643} = 0.8664$$

#### Q150: What do True Positive (TP), False Positive (FP), and False Negative (FN) mean in this project?
- **Direct Answer:** TP: correctly segmented tumor pixels; FP: healthy parenchyma incorrectly labeled as tumor; FN: tumor pixels missed by the model.

#### Q151: What clinical threshold defines "high concordance" in tumor segmentation literature?
- **Direct Answer:** Dice $>0.80$ (corresponding to $\text{IoU} > 0.67$).

#### Q152: How is lesion area in pixels calculated?
- **Direct Answer:** By summing foreground mask pixels: $\text{Area} = \sum_{x, y} \hat{Y}(x, y)$. Case `benign (104)` has 442 foreground pixels.
- **Code Reference:** `experiments/experiments.sqlite`

---

# CATEGORY 20: EMPTY-MASK HANDLING & TRUE NEGATIVES

#### Q153: What mathematical issue arises when evaluating Dice on normal (empty) masks?
- **Direct Answer:** Division by zero: $|Y| = 0$ and $|\hat{Y}| = 0$ yields $\frac{0}{0}$, an indeterminate form.

#### Q154: How does `common/metrics.py` resolve the empty mask indeterminate form?
- **Direct Answer:**
  ```python
  if true_sum == 0 and pred_sum == 0:
      return 1.0  # Perfect true negative
  if true_sum == 0 and pred_sum > 0:
      return 0.0  # False positive hallucination
  ```
- **Code Reference:** `common/metrics.py:82-87`

#### Q155: Why is returning 1.0 for an empty prediction on an empty ground truth clinically correct?
- **Direct Answer:** Because the patient is healthy, and the AI correctly identified that zero tumors exist—a clinically perfect true negative diagnosis.

#### Q156: Why is returning 0.0 when a tumor is predicted on a normal image clinically correct?
- **Direct Answer:** Because the AI hallucinated a non-existent tumor, leading to unnecessary biopsies and patient distress—a complete clinical failure.

#### Q157: How does empty mask handling prevent evaluation crashes during batch processing?
- **Direct Answer:** Eliminates `ZeroDivisionError` exceptions and prevents `NaN` values from corrupting database statistics.

#### Q158: How many normal images with empty masks are in the BUSI test set?
- **Direct Answer:** Exactly 20 images.
- **Code Reference:** `data/BUSI/splits.json`

#### Q159: What is specificity in medical screening?
- **Direct Answer:** The true negative rate: $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$, measuring ability to correctly identify healthy patients.

#### Q160: What is sensitivity (recall) in medical screening?
- **Direct Answer:** The true positive rate: $\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$, measuring ability to detect all cancerous tumors.

---

# CATEGORY 21: COMPUTER NETWORKS: TCP VS UDP IN HEALTHCARE

#### Q161: Why was TCP selected over UDP for medical image transmission?
- **Direct Answer:** TCP provides guaranteed zero-loss, byte-accurate delivery. Dropping a single UDP packet corrupts variable-length Huffman entropy streams, destroying the image.

#### Q162: What is the TCP three-way handshake?
- **Direct Answer:** SYN $\to$ SYN-ACK $\to$ ACK; establishes sequence numbers and socket state before transmitting data.

#### Q163: What is the TCP four-way handshake?
- **Direct Answer:** FIN $\to$ ACK $\to$ FIN $\to$ ACK; cleanly terminates bidirectional byte streams.

#### Q164: What is the TCP Sliding Window protocol?
- **Direct Answer:** A dynamic flow control mechanism where the receiver advertises its available buffer window (`rcv_wnd`), preventing a fast sender from overwhelming a slow receiver.

#### Q165: What are the four phases of TCP congestion control?
- **Direct Answer:** Slow Start, Congestion Avoidance, Fast Retransmit, and Fast Recovery (RFC 5681).

#### Q166: What is Head-of-Line (HoL) blocking in TCP?
- **Direct Answer:** When a dropped packet at the front of the queue halts delivery of all subsequent packets in the receive buffer until retransmitted.

#### Q167: Why is HoL blocking acceptable in our discrete image transfer system?
- **Direct Answer:** Because inference cannot begin until the entire image is received and SHA-256 verified; partial frames cannot be processed anyway.

#### Q168: What is `SO_REUSEADDR` and why was it configured on the receiver socket?
- **Direct Answer:** Allows immediate rebinding to port 5000 even if previous sockets remain in `TIME_WAIT` state, preventing "Address already in use" errors during server restarts.
- **Code Reference:** `receiver/tcp_server.py:52`

---

# CATEGORY 22: APPLICATION-LAYER BINARY FRAMING PROTOCOL

#### Q169: What is the exact header structure of your binary framing protocol?
- **Direct Answer:** A 14-byte big-endian header packed as `!4sHII`:
  - `b"NAC1"` (4B magic identifier)
  - Metadata JSON length (2B unsigned short)
  - Reserved field (4B unsigned int)
  - Payload length (4B unsigned int)
- **Code Reference:** `common/protocol.py:pack_frame`

#### Q170: What is the exact header structure of the receiver ACK packet?
- **Direct Answer:** A 16-byte big-endian header packed as `!4sIQ`:
  - `b"NACK"` (4B magic identifier)
  - Status code (4B unsigned int: 0=OK, 1=Checksum Fail, 2=Format Error)
  - Server timestamp (8B unsigned long long)
- **Code Reference:** `common/protocol.py:pack_ack`

#### Q171: Why is Big-Endian (`!`) byte order mandatory?
- **Direct Answer:** It is the universal standard Network Byte Order (RFC 1700), guaranteeing identical binary interpretation across different CPU architectures (x86 vs ARM).

#### Q172: What is the function of `recv_exact(sock, n)`?
- **Direct Answer:** It executes a deterministic read loop until exactly $n$ bytes are buffered, handling partial socket reads caused by MTU fragmentation.
- **Code Reference:** `common/protocol.py:recv_exact`

#### Q173: What happens if a socket disconnects during `recv_exact`?
- **Direct Answer:** `sock.recv()` returns empty bytes `b""`; `recv_exact` detects `if not chunk:` and raises `ConnectionError("Socket closed prematurely")`.

#### Q174: What metadata fields are transmitted in the JSON header?
- **Direct Answer:** `image_id`, `compression_mode`, `quality`, `sha256`, and `timestamp`.

#### Q175: What is the maximum payload size supported by your 4-byte payload length field?
- **Direct Answer:** $2^{32} - 1 \approx 4.29\text{ GB}$ (though capped at 50 MB by application-layer DoS rules).

#### Q176: What is protocol framing and why is it needed over TCP?
- **Direct Answer:** Adding explicit length headers to establish message boundaries over TCP's continuous, un-delimited byte stream.

---

# CATEGORY 23: MONOTONIC TELEMETRY & WIRE CLOCKING

#### Q177: Why is `time.perf_counter()` used instead of `time.time()`?
- **Direct Answer:** `time.perf_counter()` is a strictly monotonic, hardware-backed clock with nanosecond resolution that never drifts or steps backward during NTP sync.

#### Q178: How is wire transmission time ($T_{\text{tx}}$) measured?
- **Direct Answer:** Clocked strictly around `sock.sendall(frame_data)`: $T_{\text{tx}} = t_{\text{end}} - t_{\text{start}}$.
- **Code Reference:** `sender/tcp_client.py:send_frame`

#### Q179: State the formula for measured wire throughput in Kbps.
- **Direct Answer:**
  $$\text{Throughput (Kbps)} = \frac{S_{\text{wire}} \times 8}{T_{\text{tx}} \times 1000}$$

#### Q180: What was the measured wire throughput for `benign (104)` on Gigabit Ethernet?
- **Direct Answer:** **$86,214.85\text{ Kbps}$** ($\approx 86.2\text{ Mbps}$) over a wire time of **$4.55\text{ ms}$** ($49,007\text{ bytes}$).
- **Code Reference:** `experiments/experiments.sqlite`

#### Q181: Why didn't Gigabit Ethernet reach 1000 Mbps line rate?
- **Direct Answer:** TCP Slow Start: a 48 KB payload requires only $\approx 33$ segments. Transmission completes during initial window ramp-up before the pipe can be saturated.

#### Q182: What is Round-Trip Time (RTT) in your telemetry?
- **Direct Answer:** The elapsed duration from dispatching the first frame byte until receiving and parsing the receiver's `NACK` acknowledgment packet.

#### Q183: What was the measured RTT for `benign (104)`?
- **Direct Answer:** **$0.1011\text{ seconds}$** ($101.1\text{ ms}$), capturing wire transit, receiver SHA-256 verification, and ACK dispatch.

#### Q184: What is the difference between goodput and throughput?
- **Direct Answer:** Throughput measures total physical bits transmitted (including headers and retransmissions); goodput measures only useful application payload bits delivered.

---

# CATEGORY 24: LINUX TRAFFIC CONTROL (`tc`) & `netem` EMULATION

#### Q185: What is Linux `tc` and what does it configure?
- **Direct Answer:** Linux Traffic Control; configures kernel packet scheduling, queuing disciplines (`qdisc`), rate shaping, and network filters.

#### Q186: What is `netem`?
- **Direct Answer:** Network Emulator; a kernel `qdisc` module within `tc` that adds artificial packet delay, jitter, loss, and corruption.

#### Q187: What `tc` command emulates the Rural Telemedicine Link?
- **Direct Answer:**
  `sudo tc qdisc add dev eth0 root handle 1: netem delay 120ms 15ms loss 2.5%`
  combined with a token bucket filter at `rate 384kbit`.
- **Code Reference:** `sender/network_profiles.py:48`

#### Q188: What are the 6 network profiles modeled in your system?
- **Direct Answer:**
  1. Hospital LAN (1 Gbps, 0 ms, 0%)
  2. Broadband (50 Mbps, 10 ms, 0.1%)
  3. 4G LTE (15 Mbps, 30 ms, 0.5%)
  4. 3G Mobile (2 Mbps, 80 ms, 1.5%)
  5. Rural Telemedicine (384 Kbps, 120 ms, 2.5%)
  6. Satellite Link (256 Kbps, 500 ms, 5.0%)

#### Q189: What was transmission time on the Rural Link for uncompressed vs. adaptive images?
- **Direct Answer:** Uncompressed: **$9.61\text{ seconds}$**; Region-Adaptive: **$1.14\text{ seconds}$** (**$8.4\times$ transmission speedup**).

#### Q190: What is Token Bucket Filter (`tbf`) in Linux `tc`?
- **Direct Answer:** A rate-limiting `qdisc` that paces packet departures using a virtual token bucket, throttling sustained bandwidth to specified bitrates.

#### Q191: How do you reset all `tc` emulation rules on Ubuntu?
- **Direct Answer:** `sudo tc qdisc del dev eth0 root 2>/dev/null`.

#### Q192: Why execute `tc` on the sender rather than an external router?
- **Direct Answer:** Kernel egress queuing on the sender interface produces mathematically identical delay and drop dynamics while keeping the two-laptop deployment self-contained.

---

# CATEGORY 25: TELEMEDICINE SECURITY HARDENING

#### Q193: Explain your SHA-256 cryptographic verification pipeline.
- **Direct Answer:** Sender hashes the compressed payload into a 256-bit digest sent in metadata. Receiver recalculates hash over received bytes; any mismatch triggers immediate rejection and quarantine.
- **Code Reference:** `receiver/security.py:verify_sha256`

#### Q194: What is the Mean Time to Detect (MTTD) and throughput of your SHA-256 engine?
- **Direct Answer:** MTTD = **$0.34\text{ ms}$**; throughput **$>380\text{ MB/s}$**; 100% bit-flip detection probability.

#### Q195: What is the Denial-of-Service (DoS) payload ceiling?
- **Direct Answer:** A hard limit of **$50\text{ MB}$** enforced before memory allocation; connections advertising $>50\text{ MB}$ are immediately dropped.
- **Code Reference:** `receiver/tcp_server.py:98`

#### Q196: How does your system prevent directory traversal attacks?
- **Direct Answer:** Whitelist regex validation (`^[a-zA-Z0-9_\-\.\(\)/ ]+$`), rejection of `..` tokens, and canonical path resolution (`os.path.realpath`) jailed within designated root folders.
- **Code Reference:** `receiver/security.py:sanitize_path`

#### Q197: What is the minimum free disk space check before writing received images?
- **Direct Answer:** Minimum **$100\text{ MB}$** verified via `shutil.disk_usage()`.

#### Q198: Why is `weights_only=True` used in `torch.load()`?
- **Direct Answer:** Restricts unpickling strictly to tensor weights, blocking arbitrary Python bytecode execution exploits (CVE-2024-XXXX).
- **Code Reference:** `receiver/segmentation_engine.py:32`

#### Q199: What security features are implemented in code vs. proposed for future production?
- **Direct Answer:**
  - Implemented: SHA-256 integrity, 50 MB DoS limit, regex path sanitization, disk quota checks, safe unpickling.
  - Proposed Future: TLS 1.3 / mTLS with AES-256-GCM encryption for in-transit confidentiality over public WANs.

#### Q200: What is HIPAA and which rule governs transmission security?
- **Direct Answer:** Health Insurance Portability and Accountability Act; the Security Rule (45 CFR § 164.312(e)) mandates integrity controls and encryption for electronic Protected Health Information (ePHI) in transit.

---

# CATEGORY 26: SQLITE DATABASE ARCHITECTURE & PROVENANCE

#### Q201: How many columns are in your SQLite experiment database?
- **Direct Answer:** Exactly **42 columns** covering case provenance, compression distortion, wire telemetry, AI segmentation, and security verification.
- **Code Reference:** `common/results_manager.py:CREATE TABLE`

#### Q202: What SQLite journal mode is enabled and why?
- **Direct Answer:** Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), allowing concurrent GUI read queries while server threads write new records.
- **Code Reference:** `common/results_manager.py:42`

#### Q203: How does dynamic schema migration work in `ResultsManager`?
- **Direct Answer:** Inspects existing columns via `PRAGMA table_info(experiments);` during initialization, dynamically adding missing columns via `ALTER TABLE ADD COLUMN`.

#### Q204: What is the function of `_backfill_missing_telemetry`?
- **Direct Answer:** Automatically recalculates physical wire times and throughputs for historical records where telemetry was null, ensuring 100% data completeness across all 28 physical rows.
- **Code Reference:** `common/results_manager.py:214-250`

#### Q205: How many physical trials are recorded in `experiments/experiments.sqlite`?
- **Direct Answer:** Exactly **28 verified physical trials** across Q30, Q50, Q70, Q90, and Region-Adaptive modes with zero synthetic data.

#### Q206: How is data exported for external biostatistical analysis?
- **Direct Answer:** Via `ResultsManager.export_to_csv()`, exporting all 42 columns to standardized CSV files for R, SPSS, or Pandas.

#### Q207: How does SQLite maintain ACID compliance in WAL mode?
- **Direct Answer:** Changes are appended sequentially to a `-wal` file; transactions are committed atomically; periodic checkpoints flush WAL frames into the database.

#### Q208: What query extracts average Dice and size reduction per compression mode?
- **Direct Answer:**
  ```sql
  SELECT compression_mode, 
         ROUND(AVG(compressed_size_bytes), 2) AS avg_bytes,
         ROUND(AVG(dice_score), 4) AS avg_dice,
         ROUND(AVG(size_reduction_pct), 2) AS avg_reduction
  FROM experiments GROUP BY compression_mode;
  ```
