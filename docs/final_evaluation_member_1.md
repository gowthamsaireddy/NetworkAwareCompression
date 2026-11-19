# Final Project Evaluation Study Guide — Member 1
## Domain: Dataset, Digital Image Processing, Standard Compression & Image Quality Metrics

---

### Executive Profile: Member 1
- **Role:** Digital Image Processing & Compression Lead
- **Assigned Modules:**
  - BUSI Dataset ingestion, preprocessing, mask merging, and split management (`common/dataset.py`, `data/BUSI/splits.json`)
  - Standard JPEG uniform compression pipeline (`common/compression.py`)
  - Objective image distortion and similarity evaluation (`common/metrics.py`)
  - Sender GUI dataset viewer and compression workbench (`ui/sender/compression_view.py`, `ui/sender/dataset_view.py`)
- **Key Files Owned:**
  - `common/dataset.py` (Dataset loader, multi-mask union, deterministic seed 42 splitting)
  - `common/compression.py` (`compress_jpeg_uniform`, `compute_compression_stats`)
  - `common/metrics.py` (`compute_mse`, `compute_psnr`, `compute_ssim`)
  - `tests/test_compression.py` (Test suite for JPEG compression ratio, monotonic size scaling, and PSNR/SSIM)

---

# SECTION 1: FIVE-LEVEL EXPLANATION FRAMEWORK

To excel in the evaluation, you must adjust your answer's technical depth depending on who is asking. Practice explaining every core concept at these five distinct levels:

```
[Level 1: Beginner]    --> Plain English, real-world analogies, high-level intuition (for non-specialist examiners)
[Level 2: Technical]   --> Algorithmic, mathematical, engineering principles (standard viva answer)
[Level 3: Code-Level]  --> Exact repository paths, function signatures, data types, parameter choices
[Level 4: Viva Pitch]  --> 30-to-60 second rapid-fire response: structured, confident, direct
[Level 5: Defense]     --> Deep counter-examination defense against tricky "trap" questions from senior professors
```

---

## 1.1 Breast Ultrasound Imaging & The Core Clinical Problem

### Level 1: Beginner / Intuitive
"Imagine a doctor in a rural clinic examining a woman with a suspected breast lump. The clinic has a basic ultrasound machine but no specialist radiologist. To get an expert opinion, the ultrasound scan must be sent over the internet to a major city hospital. But rural internet is slow, unstable, and bandwidth-constrained. If we send the raw image, it takes too long or fails to upload. If we compress the image heavily using standard techniques to save bandwidth, the doctor at the other end receives a blurry image where the lesion's delicate boundary is corrupted. Our project solves this paradox by balancing network constraints with diagnostic accuracy."

### Level 2: Technical / Engineering
Breast Ultrasound (BUS) imaging is a non-invasive, non-ionizing, cost-effective modality widely used for detecting and classifying breast neoplasms (benign vs. malignant). However, B-mode ultrasound is characterized by low contrast, acoustic shadowing, and high-frequency speckle noise caused by constructive and destructive interference of backscattered acoustic waves. In telemedicine transmission across bandwidth-constrained channels (e.g., 2G/3G/4G or satellite links in remote healthcare settings), raw uncompressed DICOM/PNG images incur excessive transmission latency and packet loss. Applying aggressive uniform lossy compression (such as JPEG at low quality factors) introduces spatial artifacts—specifically block boundary discontinuities ($8 \times 8$ grid artifacts) and high-frequency ringing—that distort the lesion's contour, directly degrading downstream computerized lesion segmentation models.

### Level 3: Project-Specific Implementation
- **Data Source:** Dataset of Breast Ultrasound Images (BUSI) collected in 2018 from 600 female patients aged 25–75 at Baheya Hospital, Cairo.
- **Repository Manifest:** `common/dataset.py` defines `BUSIDataset` which scans `data/BUSI/` for `.png` image files and corresponding binary masks (`*mask.png`, `*mask_1.png`).
- **Input Dimensions:** Raw BUSI scans vary from $354 \times 415$ to $820 \times 748$ pixels in 8-bit grayscale / 24-bit BGR. In our standardized pipeline, inference inputs are resized to $256 \times 256 \times 1$ normalized to $[0.0, 1.0]$.
- **Clinical Motivation:** Preserve diagnostic morphology (lesion boundaries, spiculation, acoustic shadows) while drastically minimizing transmitted byte payload over TCP.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "In our project, we address the conflict between limited telemedical network bandwidth and clinical diagnostic fidelity. Breast ultrasound images contain dense acoustic speckle noise and subtle lesion margins. Standard lossy compression treats all pixels identically, discarding high-frequency edge information in the lesion to achieve a target bitrate. We systematically evaluate standard JPEG across four standard quality levels (Q30, Q50, Q70, Q90) to measure the exact threshold where lossy artifacts begin degrading deep learning segmentation accuracy, providing the empirical foundation for our region-adaptive compression engine."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why not simply transmit the images losslessly using PNG, FLAC, or JPEG-LS? Medical standards strictly forbid lossy compression!"*
- **Defense Response:**
  "While DICOM PS 3.5 allows lossless compression (like JPEG Lossless or JPEG 2000 Part 1 reversible), in rural emergency triage, disaster relief, or mobile ambulance telemetry, bandwidth is frequently constrained below 1 Mbps with 5–10% packet drop rates. Under a 384 Kbps 3G uplink, a lossless 500 KB ultrasound scan requires over 10.4 seconds of wire transfer time, excluding TCP congestion backoff and handshakes. If a clinician needs immediate triage or edge AI processing, this latency is unacceptable. Furthermore, FDA and ACR guidelines do permit irreversible compression for secondary review and triage provided diagnostic efficacy is clinically validated. Our project quantitatively proves that by isolating the lesion Region of Interest (ROI) and preserving it at Q90 while aggressively compressing the non-diagnostic background at Q20, we achieve a 37.9% byte reduction over uniform Q90 with 0.00% degradation in U-Net segmentation Dice score ($0.8664$ vs $0.8664$)."

---

## 1.2 The BUSI Dataset & Preprocessing Pipeline

### Level 1: Beginner / Intuitive
"The BUSI dataset is our digital library of 780 ultrasound scans divided into three folders: benign (non-cancerous lumps like cysts or fibroadenomas), malignant (cancerous tumors), and normal (healthy scans without lumps). For every scan with a lump, an expert radiologist hand-drew a black-and-white mask showing exactly where the lump is. If an image had two separate lumps, the dataset had two separate mask files. Our first job in code was to combine these separate masks into one master mask and split the dataset cleanly so the AI never cheats on its test."

### Level 2: Technical / Engineering
The Breast Ultrasound Image (BUSI) dataset consists of 780 images across 3 diagnostic categories:
- **Benign:** 437 images (smooth, well-circumscribed oval/round hypoechoic masses)
- **Malignant:** 210 images (irregular, ill-defined, spiculated hypoechoic lesions with posterior acoustic shadowing)
- **Normal:** 133 images (healthy breast parenchymal tissue without focal lesions; mask is all black / zeros)
Total: $437 + 210 + 133 = 780$ cases.

A critical preprocessing requirement is multi-mask handling. Certain benign and malignant cases exhibit multifocal lesions with multiple ground-truth mask files (e.g., `benign (100).png`, `benign (100)_mask.png`, `benign (100)_mask_1.png`). In `common/dataset.py`, we execute a logical bitwise-OR union ($\mathbf{M} = \bigcup_{k} \mathbf{M}_k$) implemented via `np.maximum(mask, current_mask)` to construct a single unified ground-truth binary mask per case.

### Level 3: Project-Specific Implementation
- **File:** `common/dataset.py:BUSIDataset`
- **Train/Val/Test Split:**
  - Split Ratio: 70% Training (546 images), 15% Validation (117 images), 15% Test (117 images).
  - Implementation: Stratified random sampling using `train_test_split(..., stratify=labels, random_state=42)`.
  - Storage: Saved deterministically to `data/BUSI/splits.json` to prevent train/test contamination across multiple development runs.
- **Breakdown of Test Set (117 images):**
  - Benign: 66 images
  - Malignant: 31 images
  - Normal: 20 images
- **Interpolation Methods (`cv2.resize`):**
  - Image Resizing: `cv2.INTER_LINEAR` (Bilinear interpolation) maps continuous grayscale gradients smoothly.
  - Mask Resizing: `cv2.INTER_NEAREST` (Nearest neighbor interpolation). Bilinear resizing on binary $\{0, 255\}$ masks creates intermediate fractional intensities ($0 < x < 255$), corrupting discrete boundary labels. Nearest neighbor strictly preserves $\{0, 1\}$ or $\{0, 255\}$ label integrity.

```python
# Code Reference: common/dataset.py:48-62
for mask_file in mask_files:
    m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if m is not None:
        if combined_mask is None:
            combined_mask = m
        else:
            combined_mask = np.maximum(combined_mask, m)
```

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "We use the benchmark BUSI dataset comprising 780 images: 437 benign, 210 malignant, and 133 normal cases. Because ultrasound pathology can present multiple lesions in a single scan, our dataset loader in `common/dataset.py` aggregates multiple mask files using an element-wise maximum operation (`np.maximum`). We partition the dataset into 70% training (546), 15% validation (117), and 15% held-out test (117) using stratified sampling fixed at seed 42. Crucially, image inputs use bilinear interpolation, while segmentation masks are resized strictly with nearest-neighbor interpolation to preserve binary ground-truth boundaries."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"How did you handle the 133 'normal' images? When an image has no lesion, its mask is entirely zero. How does this affect your Dice and IoU calculations?"*
- **Defense Response:**
  "In clinical screening, distinguishing between healthy tissue and neoplasms is vital. In `common/metrics.py`, when evaluating an entirely negative (empty) ground-truth mask ($|Y| = 0$):
  If the model also correctly predicts zero foreground pixels ($|\hat{Y}| = 0$), the true negative segmentation is clinically perfect. Standard formulas $\frac{2|X \cap Y|}{|X| + |Y|}$ produce $\frac{0}{0}$, an indeterminate form. We explicitly handle this in `compute_dice_score`:
  ```python
  if true_sum == 0 and pred_sum == 0:
      return 1.0  # Perfect true negative
  if true_sum == 0 and pred_sum > 0:
      return 0.0  # False positive hallucination
  ```
  This prevents division-by-zero crashes while penalizing models that hallucinate lesions in healthy scans."

---

## 1.3 Digital Image Processing (DIP) Foundations in Ultrasound

### Level 1: Beginner / Intuitive
"An ultrasound image is basically an acoustic shadow map. The machine sends high-frequency sound waves into the body and listens to the echoes. Dense tissue bounces waves back as bright white pixels; fluid cysts let sound pass through, showing up pitch black. But as sound waves bounce around, they collide and create a grainy salt-and-pepper pattern called speckle noise. In computer memory, this is stored as an 8-bit grid where 0 is pure black and 255 is pure white."

### Level 2: Technical / Engineering
From an image processing perspective, B-mode ultrasound is a 2D intensity map $I(x, y) \in [0, 255]$ generated from radiofrequency (RF) acoustic reflections:
1. **Dynamic Range & Bit Depth:** 8-bit unsigned integer (`uint8`), providing 256 discrete gray levels.
2. **Speckle Noise Modeling:** Speckle is multiplicative noise, not additive Gaussian noise. It obeys Rayleigh or Gamma distribution:
   $$I(x, y) = S(x, y) \cdot \eta(x, y)$$
   where $S(x, y)$ is the underlying tissue acoustic impedance and $\eta(x, y)$ is the stochastic interference pattern.
3. **Contrast-to-Noise Ratio (CNR):** Breast tumors typically exhibit hypoechogenicity (lower acoustic impedance than surrounding glandular tissue, creating low CNR).
4. **Morphological Filtering:** In `sender/adaptive_compressor.py`, structuring elements (`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)`) and morphological dilation expand detected lesion regions to encapsulate diagnostic margins.

### Level 3: Project-Specific Implementation
- **Array Normalization:** Images are read via `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` or converted from BGR via `cv2.COLOR_BGR2GRAY`. Prior to tensor feeding, pixel intensities are transformed to floating point:
  $$I_{\text{norm}}(x, y) = \frac{I(x, y)}{255.0} \in [0.0, 1.0]$$
- **Morphological Margin Dilation:** In `sender/adaptive_compressor.py`, when the user or ROI detector selects an expansion margin $m$ (default: 15 pixels), the mask is dilated:
  ```python
  kernel_size = 2 * margin_px + 1
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
  dilated_mask = cv2.dilate(roi_mask, kernel, iterations=1)
  ```
  An elliptical kernel is chosen over a rectangular kernel because natural biological neoplasms grow radially rather than along rectilinear grid axes.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "Ultrasound images present unique challenges in Digital Image Processing. Unlike natural photographs, ultrasound images have low contrast and are dominated by multiplicative acoustic speckle noise. They are represented as 8-bit grayscale matrices where hypoechoic lesions appear as dark regions against fibro-glandular backgrounds. In our pipeline, we employ precision morphological operations—specifically elliptical structuring element dilation—to expand the lesion boundary by a configurable 15-pixel margin, ensuring that the surrounding parenchymal transition zone is fully protected from compression degradation."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why did you use morphological dilation to create the margin instead of simply placing a rectangular bounding box around the lesion?"*
- **Defense Response:**
  "A rectangular bounding box is computationally trivial, but it has severe clinical disadvantages. Rectangular boxes include unnecessary background corner pixels that inflate the byte size of the high-quality region. More importantly, rectangular cutouts introduce four sharp, linear, $90^\circ$ edges into the image. When compressed, these artificial linear edges generate high-frequency Fourier components that cause Gibbs ringing and false edge activations in deep convolutional layers. By using morphological dilation with an elliptical kernel (`cv2.MORPH_ELLIPSE`), we preserve the organic, continuous shape of the tumor boundary while conserving bandwidth."

---

## 1.4 Standard JPEG Compression Architecture

### Level 1: Beginner / Intuitive
"JPEG is like packing a suitcase where you throw away the clothes you are least likely to wear. Human eyes notice changes in brightness much more than changes in subtle, fast textures. JPEG splits an image into tiny $8 \times 8$ pixel squares. It uses a mathematical tool called the Discrete Cosine Transform to convert those pixels into frequencies. Then, it aggressively rounds off and deletes the high-frequency details (which look like fine noise) while keeping the low-frequency shapes. The quality number (from 1 to 100) controls how aggressively it throws details away."

### Level 2: Technical / Engineering
Standard JPEG (ISO/IEC 10918-1) is a transform-based lossy image compression algorithm consisting of five sequential stages:

```
[Input Grayscale Image]
         │
         ▼
[1. 8×8 Block Partitioning]
         │
         ▼
[2. Forward 2D-DCT]  --> Converts spatial intensities to frequency coefficients
         │
         ▼
[3. Quantization]     --> Division by Quantization Matrix Q(u,v) scaled by Quality Factor
         │               (THE ONLY LOSSY STEP IN THE PIPELINE)
         ▼
[4. Zigzag Reordering] --> Clusters low-frequency DC/AC coefficients before high-frequency zeros
         │
         ▼
[5. Entropy Coding]    --> Run-Length Encoding (RLE) on zeros + Huffman Encoding
         │
         ▼
[Binary JPEG Byte Stream]
```

#### Mathematical Formulation of 2D-DCT:
For an $8 \times 8$ block of pixels $f(x, y)$ shifted by $-128$ into $[-128, 127]$, the 2D Discrete Cosine Transform (DCT-II) is defined as:
$$F(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left[ \frac{(2x + 1)u\pi}{16} \right] \cos\left[ \frac{(2y + 1)v\pi}{16} \right]$$
where $u, v \in \{0, 1, \dots, 7\}$ are spatial frequency indices, and:
$$C(u) = \begin{cases} \frac{1}{\sqrt{2}} & \text{for } u = 0 \\ 1 & \text{for } u > 0 \end{cases}$$
- $F(0, 0)$ is the **DC coefficient**, representing $8 \times$ the average intensity of the $8 \times 8$ block.
- The remaining 63 coefficients are **AC coefficients**, representing increasingly higher horizontal, vertical, and diagonal spatial frequencies.

#### Quantization Matrix Scaling:
Quantization is the sole irreversible, lossy step:
$$F_Q(u, v) = \text{round}\left( \frac{F(u, v)}{Q(u, v)} \right)$$
The standard JPEG luminance quantization table $Q_{50}$ is scaled according to user Quality Factor $Q \in [1, 100]$:
$$S = \begin{cases} \frac{5000}{Q} & \text{if } Q < 50 \\ 200 - 2Q & \text{if } Q \ge 50 \end{cases}$$
$$Q_{\text{scaled}}(u, v) = \text{clip}\left( \text{round}\left( \frac{Q_{50}(u, v) \cdot S + 50}{100} \right), 1, 255 \right)$$

### Level 3: Project-Specific Implementation
- **File:** `common/compression.py:compress_jpeg_uniform`
- **Implementation:** Leverages OpenCV's libjpeg backend:
  ```python
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
  success, encoded_img = cv2.imencode('.jpg', image, encode_param)
  compressed_bytes = encoded_img.tobytes()
  ```
- **Tested Quality Levels:**
  - **Q90:** High fidelity. Artifacts invisible to the human eye. Compression Ratio $\approx 5:1$ to $9:1$.
  - **Q70:** Moderate compression. Compression Ratio $\approx 10:1$ to $15:1$. Minor high-frequency attenuation.
  - **Q50:** Noticeable $8 \times 8$ block boundary discontinuities begin to appear. Compression Ratio $\approx 16:1$ to $20:1$.
  - **Q30:** Aggressive compression. Severe blockiness, ringing around sharp edges, loss of low-contrast lesions. Compression Ratio $\approx 25:1$ to $35:1$.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "Standard JPEG compresses images through five core stages: $8 \times 8$ block partitioning, 2D Discrete Cosine Transform, scalar quantization, zigzag reordering, and Huffman entropy coding. Quantization is the only lossy stage in the entire pipeline, where DCT coefficients are divided by a scaled quantization table and rounded to integers. In our project, we evaluate four discrete quality factors: Q90, Q70, Q50, and Q30. At Q30, aggressive quantization forces high-frequency AC coefficients to zero, creating visible $8 \times 8$ blocking artifacts and blurring subtle acoustic margins."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why does JPEG compression affect ultrasound images much worse than natural camera photos?"*
- **Defense Response:**
  "Natural photographs are characterized by smooth intensity gradients and low spatial frequencies; natural camera noise is generally low-amplitude and additive. In contrast, medical ultrasound images contain pervasive acoustic speckle noise, which is inherently a high-frequency phenomenon. When JPEG computes the 2D-DCT of an ultrasound block, the speckle noise populates the high-frequency AC coefficients. When quantized at Q30 or Q50, these high-frequency coefficients are truncated to zero. This does two catastrophic things: first, it strips away the textural speckle pattern that radiologists use to discern tissue stiffness; second, it causes severe contrast mismatches across block edges, generating artificial rectilinear boundaries that downstream convolutional kernels mistake for anatomical edges."

---

## 1.5 Objective Image Quality Metrics (MSE, PSNR, SSIM)

### Level 1: Beginner / Intuitive
"When we compress an image, we degrade it. How do we measure how damaged it is? We use three scorecards:
1. **MSE (Mean Squared Error):** The average squared pixel error. 0 means identical; higher numbers mean more corruption.
2. **PSNR (Peak Signal-to-Noise Ratio):** Measured in decibels (dB), like sound. Above 35 dB is excellent; below 30 dB is visibly degraded.
3. **SSIM (Structural Similarity Index):** Scored from 0 to 1, mimicking how the human eye perceives brightness, contrast, and structure. 1.0 means a 100% perfect structural match."

### Level 2: Technical / Engineering

#### 1. Mean Squared Error (MSE):
For an uncompressed image $I$ and a reconstructed compressed image $K$ of dimensions $M \times N$:
$$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I(x, y) - K(x, y)]^2$$

#### 2. Peak Signal-to-Noise Ratio (PSNR):
PSNR computes the ratio between the maximum possible signal power ($MAX_I^2 = 255^2 = 65025$) and the corrupting noise power (MSE), expressed on a logarithmic decibel scale:
$$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{MAX_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10}(255) - 10 \cdot \log_{10}(\text{MSE}) = 48.1308 - 10 \cdot \log_{10}(\text{MSE})$$
- If $\text{MSE} = 0$ (identical images), $\text{PSNR} \to \infty$. In our code, we return `100.0 dB` to avoid floating-point overflow.

#### 3. Structural Similarity Index (SSIM):
While MSE and PSNR measure absolute pixel differences, they fail to reflect perceived structural fidelity. SSIM evaluates luminance $l(x, y)$, contrast $c(x, y)$, and structural correlation $s(x, y)$ over an $11 \times 11$ Gaussian sliding window:
$$\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma$$
Setting $\alpha = \beta = \gamma = 1$:
$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
Where:
- $\mu_x, \mu_y$ are local window means.
- $\sigma_x^2, \sigma_y^2$ are local sample variances.
- $\sigma_{xy}$ is the sample covariance.
- $C_1 = (K_1 L)^2$ and $C_2 = (K_2 L)^2$ are numerical stability constants, with $L = 255$, $K_1 = 0.01$ ($C_1 = 6.5025$), and $K_2 = 0.03$ ($C_2 = 58.5225$).

### Level 3: Project-Specific Implementation
- **File:** `common/metrics.py`
  - `compute_mse(orig, comp)`: NumPy vectorization `np.mean((orig.astype(np.float64) - comp.astype(np.float64)) ** 2)`.
  - `compute_psnr(orig, comp)`: Handles `mse == 0` check; returns `min(100.0, 10 * np.log10(65025.0 / mse))`.
  - `compute_ssim(orig, comp)`: Uses `skimage.metrics.structural_similarity` with `data_range=255`.
  - `compute_roi_psnr(orig, comp, roi_mask)`: Masks pixels outside the dilated ROI before computing MSE and PSNR.
- **Physical Verification Data from `experiments/experiments.sqlite`:**
  - Case `benign (107)` at Q90: $\text{PSNR} = 74.57\text{ dB}$, $\text{SSIM} = 1.0000$.
  - Case `benign (107)` at Q70: $\text{PSNR} = 43.90\text{ dB}$, $\text{SSIM} = 0.9850$.
  - Case `benign (107)` at Q50: $\text{PSNR} = 41.16\text{ dB}$, $\text{SSIM} = 0.9636$.
  - Case `benign (107)` at Q30: $\text{PSNR} = 38.57\text{ dB}$, $\text{SSIM} = 0.9406$.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "We measure compression fidelity using MSE, PSNR, and SSIM. MSE computes the average squared pixel error, from which PSNR is derived logarithmically relative to the peak 8-bit dynamic range of 255. Because PSNR measures point-to-point error rather than visual structure, we also calculate the Structural Similarity Index (SSIM), which models luminance, contrast, and structural correlation using Gaussian sliding windows with stabilizing constants $C_1 = 6.5025$ and $C_2 = 58.5225$. Furthermore, we implement a custom ROI-PSNR metric that isolates the error specifically inside the lesion boundary."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why does an image with a high PSNR sometimes yield a lower segmentation Dice score than an image with lower PSNR?"*
- **Defense Response:**
  "PSNR is a global spatial average that treats all pixel errors equally. If an image undergoes uniform compression, 95% of the pixels belong to non-diagnostic background parenchymal tissue. If the background is preserved well, the global PSNR will remain deceptively high (e.g., $>38\text{ dB}$). However, if the 5% of pixels corresponding to the subtle, low-contrast lesion margin suffer severe high-frequency attenuation, the convolutional filters in the U-Net will fail to detect the lesion contour. This is the exact limitation of global PSNR that motivated our region-adaptive approach and our dedicated ROI-PSNR evaluation metric."

---

# SECTION 2: WORKED NUMERICAL EXAMPLES

Examiners love asking for manual calculations on paper or the whiteboard. Practice these exact calculations:

### Example 1: Manual Calculation of MSE and PSNR
Suppose we have a $2 \times 2$ patch of uncompressed pixels $I$ and compressed pixels $K$:
$$I = \begin{bmatrix} 100 & 150 \\ 200 & 50 \end{bmatrix}, \quad K = \begin{bmatrix} 102 & 148 \\ 195 & 54 \end{bmatrix}$$

**Step 1: Compute absolute differences $(I - K)$:**
$$I - K = \begin{bmatrix} -2 & +2 \\ +5 & -4 \end{bmatrix}$$

**Step 2: Compute squared differences $(I - K)^2$:**
$$(I - K)^2 = \begin{bmatrix} 4 & 4 \\ 25 & 16 \end{bmatrix}$$

**Step 3: Calculate Mean Squared Error (MSE):**
$$\text{MSE} = \frac{4 + 4 + 25 + 16}{4} = \frac{49}{4} = 12.25$$

**Step 4: Calculate PSNR:**
$$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}} \right) = 10 \cdot \log_{10}\left( \frac{65025}{12.25} \right) = 10 \cdot \log_{10}(5308.163)$$
Since $\log_{10}(5308.163) \approx 3.7249$:
$$\text{PSNR} = 10 \times 3.7249 = 37.25\text{ dB}$$

---

### Example 2: Compression Ratio (CR) and Percentage Size Reduction
An uncompressed ultrasound scan `benign (104)` has raw byte size $S_{\text{orig}} = 455,471\text{ bytes}$ ($444.8\text{ KB}$).
When compressed under Region-Adaptive compression, its size becomes $S_{\text{comp}} = 48,481\text{ bytes}$ ($47.34\text{ KB}$).

**Step 1: Compute Compression Ratio (CR):**
$$\text{CR} = \frac{S_{\text{orig}}}{S_{\text{comp}}} = \frac{455,471}{48,481} = 9.3948 \approx 9.39:1$$

**Step 2: Compute Percentage Size Reduction:**
$$\text{Reduction } \% = \left( 1 - \frac{S_{\text{comp}}}{S_{\text{orig}}} \right) \times 100\% = \left( 1 - \frac{48,481}{455,471} \right) \times 100\% = (1 - 0.10644) \times 100\% = 89.36\%$$

---

# SECTION 3: PRESENTATION SCRIPTS & HANDOVERS

## 3.1 Five-Minute Presentation Script (Member 1)

> "Respected members of the evaluation committee, good morning. I am Member 1, responsible for Dataset Ingestion, Digital Image Processing, and Standard Image Compression.
> 
> In telemedicine, the primary bottleneck in transmitting medical ultrasound scans from peripheral clinics to tertiary hospitals is limited network bandwidth. While lossy compression reduces transmission payload, conventional compression treats all pixels uniformly.
> 
> To study this, our team utilized the benchmark BUSI dataset containing 780 ultrasound images divided into 437 benign, 210 malignant, and 133 normal cases. As part of my module, I implemented a robust dataset management pipeline in `common/dataset.py`. We encountered cases where patients presented multiple lesions; my code performs an automated morphological union using an element-wise maximum operation to generate unified ground-truth masks.
> 
> We split the dataset into 70% train (546 images), 15% validation (117 images), and 15% test (117 images) using stratified sampling fixed at seed 42 to strictly prevent data leakage. During resizing, my pipeline applies bilinear interpolation to grayscale ultrasound scans, while strictly enforcing nearest-neighbor interpolation on binary masks to prevent fractional label corruption.
> 
> In our baseline compression pipeline in `common/compression.py`, I implemented uniform JPEG compression across four discrete clinical quality levels: Q90, Q70, Q50, and Q30. Through the 2D Discrete Cosine Transform and scalar quantization, high-frequency coefficients are truncated. Our quantitative analysis using MSE, PSNR, and SSIM revealed that while Q90 achieves near-lossless fidelity above 70 dB PSNR, aggressive compression at Q30 causes severe $8 \times 8$ blocking artifacts and acoustic boundary blurring.
> 
> To evaluate how these compression artifacts impact automated clinical diagnosis, I now hand over the presentation to Member 2, who will explain our deep learning segmentation and region-adaptive compression architectures."

---

## 3.2 Ten-Minute Extended Presentation Script (Member 1)

> *(Use for in-depth technical presentations)*
> 
> "Respected evaluators, good morning. My name is Member 1, and I lead the Dataset, Digital Image Processing, and Standard Compression engineering for our project: *Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation*.
> 
> **1. Clinical Motivation & Telemedicine Context:**
> Breast cancer remains one of the leading causes of oncological mortality worldwide. Early detection via B-mode ultrasound is critical. However, in rural healthcare centers, expert radiologists are unavailable. Transmitting uncompressed ultrasound scans over unstable 3G/4G or satellite networks introduces significant latency, buffering, and packet drop. Uncompressed images in the BUSI dataset range up to 455 KB. While this seems modest on broadband, across an emulated 384 Kbps rural uplink with a 50 ms latency and 2% packet loss, uncompressed TCP transmission requires multiple seconds, heavily delayed by TCP congestion control and retransmissions.
> 
> **2. The BUSI Dataset & Preprocessing Pipeline:**
> We established our foundation on the landmark BUSI dataset comprising 780 clinical images. In `common/dataset.py`, I built the ingestion and ground-truth reconciliation pipeline. A unique technical challenge in BUSI is the presence of multiple mask files per image in multifocal lesion scans. To reconcile these, my loader executes a binary union across all mask instances using `np.maximum`, ensuring no tumor foci are omitted.
> 
> We partitioned the dataset into 70% training, 15% validation, and 15% testing using stratified sampling with a deterministic seed of 42. Stratification ensures that the benign, malignant, and normal distributions are identical across all three subsets. The 117 test images remained strictly quarantined and were never seen during any phase of model training.
> 
> When resizing inputs to the $256 \times 256$ inference resolution, I applied bilinear interpolation (`cv2.INTER_LINEAR`) for ultrasound scans to preserve acoustic intensity gradients, but strictly applied nearest-neighbor interpolation (`cv2.INTER_NEAREST`) for binary masks. This was essential: standard bilinear interpolation introduces intermediate gray values between 0 and 255 at lesion boundaries, which falsely inflates or deflates the ground-truth area.
> 
> **3. Standard JPEG Pipeline & Transform Mechanics:**
> In `common/compression.py`, I integrated the standard baseline compression engine utilizing the 2D Discrete Cosine Transform (DCT). The algorithm partitions the 8-bit image into $8 \times 8$ pixel blocks, maps them into spatial frequencies via DCT, divides them by the scaled quantization matrix $Q(u, v)$, and executes zigzag run-length and Huffman entropy encoding.
> 
> We evaluated four clinical quality factors: Q90, Q70, Q50, and Q30. At Q90, compression ratios hover around $6:1$ to $9:1$, maintaining pristine structural fidelity. However, as quality decreases to Q50 and Q30, the scalar quantization step zeros out virtually all high-frequency AC coefficients. Because ultrasound speckle noise is a high-frequency phenomenon, discarding these coefficients destroys the fine textural contrast at tumor margins and introduces severe $8 \times 8$ block boundary discontinuities.
> 
> **4. Mathematical Distortion Evaluation:**
> In `common/metrics.py`, I implemented our objective evaluation framework. We calculate Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and the Structural Similarity Index (SSIM) using an $11 \times 11$ Gaussian sliding window with stabilizing constants $C_1 = 6.5025$ and $C_2 = 58.5225$. Furthermore, I designed a specialized ROI-PSNR metric that isolates distortion exclusively within the lesion region.
> 
> Our empirical data confirms that while uniform Q30 achieves an impressive $89.4\%$ byte size reduction, it degrades SSIM to $0.9406$ and drops lesion boundary clarity, directly impairing downstream AI inference.
> 
> This quantitative trade-off revealed the need for an intelligent compression method that preserves the lesion while aggressively compressing the background. I now hand over to Member 2 to present our deep learning architectures, multi-objective optimization, and region-adaptive compression engine."

---

# SECTION 4: 50 DEDICATED VIVA QUESTIONS & DEFENSE (MEMBER 1)

### Q1: What does BUSI stand for, and what are its core characteristics?
- **Direct Answer:** BUSI stands for **Breast Ultrasound Images** dataset, collected in 2018 at Baheya Hospital for Early Detection & Treatment of Women's Cancer in Cairo, Egypt.
- **Technical Detail:** It contains 780 images from 600 female patients aged 25–75, acquired on a LOGIQ E9 ultrasound system. The dataset is categorized into 437 benign, 210 malignant, and 133 normal scans, accompanied by ground-truth binary lesion masks hand-delineated by expert radiologists.
- **Code Reference:** `common/dataset.py:BUSIDataset`
- **Equation:** $N_{\text{total}} = N_{\text{benign}} (437) + N_{\text{malignant}} (210) + N_{\text{normal}} (133) = 780$.

### Q2: Why did you partition the dataset into 70/15/15 rather than standard 80/20?
- **Direct Answer:** A 70% train (546), 15% validation (117), and 15% test (117) partition guarantees a sufficiently large held-out test set (117 images) to ensure statistical significance during evaluation while retaining 117 validation samples for unbiased early stopping and hyperparameter tuning.
- **Technical Detail:** An 80/20 split without an independent validation set risks validation leakage, where hyperparameters are tuned directly on test data.
- **Code Reference:** `data/BUSI/splits.json` generated with `train_test_split(..., test_size=0.15, random_state=42)`.

### Q3: What is data leakage, and how did you prevent it?
- **Direct Answer:** Data leakage occurs when information from outside the training dataset (such as validation or test set images) unintentionally influences model training.
- **Technical Detail:** In medical imaging, data leakage can occur if slices or augmented views from the same patient appear in both train and test sets, or if normalizer statistics ($\mu, \sigma$) are computed across the entire dataset prior to splitting. We prevented leakage by computing splits once with a fixed seed (`random_state=42`), saving them to `data/BUSI/splits.json`, and freezing test images completely away from ROI detector and U-Net training.

### Q4: How are multiple mask files per image handled in your code?
- **Direct Answer:** We combine multiple mask files into a single master ground-truth mask using an element-wise maximum operation.
- **Technical Detail:** Certain multifocal tumors contain secondary files like `benign (100)_mask_1.png`. In `common/dataset.py`, lines 53–58, when a secondary mask is detected, we execute `combined_mask = np.maximum(combined_mask, m)`, which acts as a boolean logical OR without allowing pixel intensities to exceed 255.
- **Code Reference:** `common/dataset.py:53-58`

### Q5: Why must you use nearest-neighbor interpolation for masks and bilinear for images?
- **Direct Answer:** Bilinear interpolation calculates weighted averages of surrounding pixels, which turns binary masks $\{0, 255\}$ into continuous fractional values ($0 < x < 255$), corrupting boundary definitions. Nearest-neighbor strictly assigns the label of the closest pixel, preserving discrete binary labels.
- **Technical Detail:** In contrast, images contain continuous acoustic gradients where nearest-neighbor would create jagged, blocky pixelation artifacts. Hence, `cv2.INTER_LINEAR` is used for images and `cv2.INTER_NEAREST` for masks.
- **Code Reference:** `common/dataset.py:102-108`

### Q6: What is acoustic speckle noise in ultrasound, and is it additive or multiplicative?
- **Direct Answer:** Speckle noise is a granular, multiplicative interference pattern caused by the constructive and destructive phase interference of backscattered acoustic waves from sub-resolution tissue scatterers.
- **Technical Detail:** It is modeled mathematically as $I(x, y) = S(x, y) \cdot \eta(x, y)$, where $\eta$ follows a Rayleigh distribution. It is not additive thermal sensor noise; it carries subtle tissue texture information.

### Q7: Explain the 5 core stages of standard JPEG compression.
- **Direct Answer:** 
  1. Color space conversion / Grayscale partitioning into $8 \times 8$ pixel blocks.
  2. Level shifting by $-128$ and forward 2D Discrete Cosine Transform (2D-DCT).
  3. Scalar quantization via division by the scaled quantization matrix $Q(u, v)$ and rounding.
  4. Zigzag reordering to group low-frequency coefficients and long runs of high-frequency zeros.
  5. Run-length encoding (RLE) and Huffman entropy coding.

### Q8: What is the Discrete Cosine Transform (DCT) formula?
- **Direct Answer:** The 2D-DCT-II maps an $8 \times 8$ spatial matrix $f(x, y)$ to frequency coefficients $F(u, v)$:
  $$F(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left[ \frac{(2x+1)u\pi}{16} \right] \cos\left[ \frac{(2y+1)v\pi}{16} \right]$$
  where $C(w) = \frac{1}{\sqrt{2}}$ for $w=0$, and $C(w) = 1$ for $w > 0$.

### Q9: What is the difference between the DC and AC coefficients in DCT?
- **Direct Answer:** The DC coefficient $F(0, 0)$ represents zero frequency—it equals 8 times the average intensity of the entire $8 \times 8$ block. The other 63 coefficients are AC coefficients, representing variations in intensity across horizontal, vertical, and diagonal spatial frequencies.

### Q10: Which specific step in JPEG compression causes information loss?
- **Direct Answer:** Quantization is the only lossy step in JPEG compression.
- **Technical Detail:** Division of DCT coefficients by quantization matrix entries followed by integer rounding ($\text{round}(F(u, v) / Q(u, v))$) truncates fractional values, permanently discarding high-frequency detail. DCT, zigzag scanning, and Huffman coding are 100% mathematically reversible.

### Q11: How is the JPEG Quantization Matrix scaled for a given Quality Factor Q?
- **Direct Answer:** Standard luminance table $Q_{50}$ is scaled by factor $S$:
  If $Q < 50$, $S = 5000 / Q$. If $Q \ge 50$, $S = 200 - 2Q$.
  The scaled table is $Q_{\text{scaled}}(u, v) = \text{clip}\left(\text{round}\left(\frac{Q_{50}(u, v) \cdot S + 50}{100}\right), 1, 255\right)$.

### Q12: Why does aggressive JPEG compression create $8 \times 8$ blocking artifacts?
- **Direct Answer:** Because each $8 \times 8$ block is transformed and quantized independently with no boundary smoothing.
- **Technical Detail:** At low bitrates, high AC coefficients are rounded to zero, reducing each block to its average DC value or a simple planar slope. Discontinuities between the independent DC averages of adjacent blocks create visible rectilinear grid lines.

### Q13: What is Mean Squared Error (MSE)? Give the formula.
- **Direct Answer:** MSE is the average of the squared intensity differences between corresponding pixels of the original image $I$ and compressed image $K$:
  $$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I(x, y) - K(x, y)]^2$$
- **Code Reference:** `common/metrics.py:compute_mse`

### Q14: What is Peak Signal-to-Noise Ratio (PSNR)? Give the formula.
- **Direct Answer:** PSNR expresses the ratio between peak signal intensity and reconstruction noise in decibels:
  $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}} \right)$$
- **Code Reference:** `common/metrics.py:compute_psnr`

### Q15: Why is PSNR measured in decibels (dB) on a logarithmic scale?
- **Direct Answer:** Because the human perceptual response to acoustic and optical stimulus intensity is logarithmic (Weber-Fechner Law), and pixel error spans multiple orders of magnitude (from $0.001$ to $>1000$).

### Q16: What does a PSNR of $\infty$ mean, and how does your code handle it?
- **Direct Answer:** A PSNR of $\infty$ occurs when $\text{MSE} = 0$, meaning the compressed image is bit-for-bit identical to the original.
- **Technical Detail:** To prevent `ZeroDivisionError` or floating-point IEEE-754 `inf` values from crashing downstream database logging, `common/metrics.py` explicitly checks `if mse == 0: return 100.0`.

### Q17: What are typical acceptable PSNR values in medical imaging?
- **Direct Answer:** In medical imaging literature, PSNR $>40\text{ dB}$ is considered excellent (visually lossless); $35\text{ dB}$ to $40\text{ dB}$ is good; $30\text{ dB}$ to $35\text{ dB}$ is marginal; and $<30\text{ dB}$ indicates unacceptable diagnostic degradation.

### Q18: What is SSIM and why is it preferred over PSNR for human visual assessment?
- **Direct Answer:** SSIM (Structural Similarity Index) measures perceived degradation by comparing local luminance, contrast, and structural correlation rather than raw pixel-by-pixel arithmetic error.
- **Technical Detail:** The human visual system is adapted to extract structural information (edges, textures) rather than absolute luminance. Two images can have identical MSE but completely different perceived structural fidelity.

### Q19: State the mathematical formula for SSIM.
- **Direct Answer:**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  Where $C_1 = (0.01 \times 255)^2 = 6.5025$ and $C_2 = (0.03 \times 255)^2 = 58.5225$.

### Q20: What are $C_1$ and $C_2$ in SSIM, and what happens if they are omitted?
- **Direct Answer:** $C_1$ and $C_2$ are small positive constants that ensure numerical stability.
- **Technical Detail:** If they were omitted ($C_1 = C_2 = 0$), in uniform black regions where local mean $\mu = 0$ and local variance $\sigma^2 = 0$, the denominator would evaluate to zero, producing division-by-zero crashes.

### Q21: What is the range of SSIM values?
- **Direct Answer:** SSIM is bounded in the interval $[-1.0, +1.0]$, where $+1.0$ indicates identical images, $0.0$ indicates no structural correlation, and negative values indicate anti-correlation. In practice on positive grayscale images, it ranges between $0.0$ and $1.0$.

### Q22: What is the Compression Ratio (CR) formula?
- **Direct Answer:**
  $$\text{CR} = \frac{\text{Uncompressed File Size in Bytes}}{\text{Compressed File Size in Bytes}} = \frac{S_{\text{orig}}}{S_{\text{comp}}}$$
- **Code Reference:** `common/compression.py:compute_compression_stats`

### Q23: What is the Percentage Size Reduction formula?
- **Direct Answer:**
  $$\text{Size Reduction } \% = \left( 1 - \frac{S_{\text{comp}}}{S_{\text{orig}}} \right) \times 100\%$$
- **Technical Detail:** If an uncompressed 100 KB image compresses to 20 KB, the CR is $5:1$ and the size reduction is $80\%$.

### Q24: What is the relationship between Bitrate (bpp) and Compression Ratio?
- **Direct Answer:** Bitrate in bits-per-pixel (bpp) is given by:
  $$\text{bpp} = \frac{S_{\text{comp}} \times 8}{M \times N} = \frac{8}{\text{CR}}$$ (for 8-bit grayscale images).

### Q25: How does your code calculate ROI-specific PSNR?
- **Direct Answer:** In `common/metrics.py`, we apply the binary dilated lesion mask as a boolean index to extract only the pixels belonging to the lesion and margin, computing MSE exclusively over those coordinates:
  $$\text{MSE}_{\text{ROI}} = \frac{1}{|\text{ROI}|} \sum_{(x, y) \in \text{ROI}} [I(x, y) - K(x, y)]^2$$
  $$\text{PSNR}_{\text{ROI}} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}_{\text{ROI}}} \right)$$

### Q26: In your physical experiment database, what was the byte size and PSNR of `benign (107)` across Q30, Q50, Q70, and Q90?
- **Direct Answer:** From `experiments/experiments.sqlite`:
  - **Q90:** 78,280 bytes, $\text{PSNR} = 74.57\text{ dB}$, $\text{SSIM} = 1.0000$
  - **Q70:** 45,585 bytes, $\text{PSNR} = 43.90\text{ dB}$, $\text{SSIM} = 0.9850$
  - **Q50:** 33,349 bytes, $\text{PSNR} = 41.16\text{ dB}$, $\text{SSIM} = 0.9636$
  - **Q30:** 24,528 bytes, $\text{PSNR} = 38.57\text{ dB}$, $\text{SSIM} = 0.9406$

### Q27: Why did `benign (107)` at Q90 show an unusually high PSNR of 74.57 dB?
- **Direct Answer:** Because at Q90, libjpeg quantization divisors for low-frequency coefficients are mostly 1, causing minimal quantization error across smooth parenchymal regions.

### Q28: What is Huffman coding in JPEG, and why is it lossless?
- **Direct Answer:** Huffman coding is a variable-length prefix entropy coding algorithm that assigns shorter binary codes to frequently occurring symbols and longer codes to rare symbols. It is completely lossless because each code is uniquely decodable via a prefix tree.

### Q29: What is Zigzag scanning and why is it used in JPEG?
- **Direct Answer:** Zigzag scanning traverses the $8 \times 8$ matrix of quantized DCT coefficients diagonally from top-left (low frequencies) to bottom-right (high frequencies). This groups the non-zero low-frequency coefficients together and clusters the long runs of zeroed high-frequency coefficients at the end, maximizing run-length compression efficiency before the End-of-Block (EOB) marker.

### Q30: What is the difference between lossy and lossless compression?
- **Direct Answer:** Lossless compression (PNG, FLAC, ZIP) allows exact bit-for-bit reconstruction of original data without loss ($\text{MSE} = 0$). Lossy compression (JPEG, MP3) discards visually or psycho-acoustically redundant data to achieve significantly higher compression ratios, preventing exact bit-level reconstruction.

### Q31: Can JPEG compression be lossless?
- **Direct Answer:** Standard DCT-based JPEG (ISO/IEC 10918-1) is always lossy due to quantization. While there is a standard known as JPEG-LS and a spatial DPCM lossless mode in JPEG 2000, standard baseline JPEG using `cv2.imencode('.jpg', ...)` is inherently lossy even at $Q=100$.

### Q32: What happens if you pass Quality = 100 to `cv2.imencode`? Is it lossless?
- **Direct Answer:** No, it is not completely lossless. At $Q=100$, the quantization table entries are clamped to 1, but floating-point rounding errors in the forward and inverse DCT, color conversion, and level shifting still result in non-zero reconstruction error ($\text{PSNR} \approx 50\text{–}60\text{ dB}$).

### Q33: Why did you use grayscale representation instead of 3-channel RGB for ultrasound?
- **Direct Answer:** B-mode medical ultrasound is fundamentally a single-channel acoustic backscatter intensity map. RGB ultrasound files are simply 3 identical replicated channels (R=G=B). Storing or transmitting them as 3-channel RGB triples the uncompressed payload ($3 \times$ memory and wire overhead) with zero diagnostic information gain.

### Q34: What is the dynamic range of an 8-bit image?
- **Direct Answer:** An 8-bit unsigned integer (`uint8`) has $2^8 = 256$ discrete intensity levels, ranging from 0 (minimum intensity, black) to 255 (maximum intensity, white).

### Q35: How does speckle noise impact JPEG quantization?
- **Direct Answer:** Speckle noise manifests as high spatial frequency variations across adjacent pixels. When transformed via 2D-DCT, it populates high-frequency AC coefficients. Standard JPEG quantization tables heavily quantize high frequencies, treating speckle as discardable noise, which can inadvertently blur fine micro-calcifications and subtle acoustic shadowing.

### Q36: What is morphological dilation? State its mathematical definition.
- **Direct Answer:** Dilation is a morphological operation that expands object boundaries in a binary image. The dilation of image $A$ by structuring element $B$ is defined as the set union of translations:
  $$A \oplus B = \{ z \in \mathbb{Z}^2 \mid (\hat{B})_z \cap A \neq \emptyset \}$$
  In code, it sets pixel $(x, y)$ to 1 if any pixel in the structuring element overlaps a foreground pixel.

### Q37: Why did you choose an elliptical structuring element (`cv2.MORPH_ELLIPSE`) over a rectangle?
- **Direct Answer:** Biological neoplasms grow radially through surrounding tissue, forming curved, organic margins. An elliptical kernel produces smooth, rounded margin expansions without sharp rectangular corners that cause artificial high-frequency Fourier components.

### Q38: What OpenCV function was used for JPEG encoding, and what are its arguments?
- **Direct Answer:** `cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])`.
- **Technical Detail:** It returns a boolean status flag and a 1D NumPy array of compressed bytes.
- **Code Reference:** `common/compression.py:28-32`

### Q39: What is the purpose of `cv2.imdecode` on the receiver?
- **Direct Answer:** `cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)` reconstructs the spatial pixel matrix from the raw in-memory JPEG byte buffer without needing to write a temporary file to disk.
- **Code Reference:** `common/compression.py:38-42`

### Q40: What happens to the file size if you compress an entirely black image versus a complex ultrasound scan at Q90?
- **Direct Answer:** An entirely black image compresses to a tiny file (under 1 KB) because all AC coefficients are zero and the DC coefficient is zero, which Huffman coding represents with a single End-of-Block (EOB) code. A complex ultrasound image compresses to 40–80 KB due to dense acoustic speckle producing hundreds of non-zero AC coefficients.

### Q41: What is the difference between spatial resolution and radiometric resolution?
- **Direct Answer:** Spatial resolution refers to the physical pixel grid density ($M \times N$, e.g., $256 \times 256$). Radiometric resolution refers to the bit depth or number of discrete gray levels each pixel can represent ($2^8 = 256$ levels for 8-bit, $2^{16} = 65536$ levels for 16-bit).

### Q42: What is the difference between lossy compression ratio and bitrate?
- **Direct Answer:** Compression Ratio is a unitless ratio of uncompressed bytes to compressed bytes ($CR = S_{\text{orig}} / S_{\text{comp}}$). Bitrate represents the average bits used per pixel ($\text{bpp} = (S_{\text{comp}} \times 8) / (M \times N)$).

### Q43: How do you verify that your test set was completely unseen during training?
- **Direct Answer:** In `data/BUSI/splits.json`, the train, val, and test file lists are explicitly disjoint ($S_{\text{train}} \cap S_{\text{test}} = \emptyset$ and $S_{\text{val}} \cap S_{\text{test}} = \emptyset$). In training scripts, only `splits['train']` and `splits['val']` paths are passed to the PyTorch DataLoaders.

### Q44: What is the effect of downsampling an ultrasound image to $256 \times 256$?
- **Direct Answer:** Downsampling standardizes the tensor input dimension for deep learning inference and reduces GPU memory overhead. However, it slightly smooths fine acoustic details. We compensate by performing all final diagnostic boundary rendering on the uncropped full-resolution image.

### Q45: In `common/metrics.py`, how are image dimensions verified before computing metrics?
- **Direct Answer:** The function asserts that `orig.shape == comp.shape`. If dimensions differ, it raises a `ValueError("Image dimensions must match for metric calculation")`, preventing misaligned matrix operations.

### Q46: What is a hypoechoic mass versus an hyperechoic mass?
- **Direct Answer:** A hypoechoic mass reflects fewer sound waves than surrounding tissue, appearing dark gray to black (typical of fluid cysts, benign fibroadenomas, and carcinomas). A hyperechoic mass reflects more sound waves, appearing bright white (typical of fibrous tissue, calcifications, and bone).

### Q47: Why is random seed 42 used consistently throughout the codebase?
- **Direct Answer:** Random seed 42 ensures complete pseudo-random determinism. Any researcher or examiner running `common/dataset.py` or model evaluation will reproduce the exact same train/val/test splits and initialization weights down to the single image.

### Q48: What is Gibbs ringing artifact in image compression?
- **Direct Answer:** Gibbs phenomenon is an oscillatory overshoot that occurs when approximating discontinuous functions using a truncated Fourier series. In JPEG, aggressive truncation of high-frequency DCT coefficients causes visible ripple/halo patterns along sharp boundaries, such as tumor-parenchyma interfaces.

### Q49: Does standard JPEG support 16-bit medical images?
- **Direct Answer:** Standard baseline JPEG (ISO/IEC 10918-1) is strictly limited to 8-bit sample precision. Extended JPEG processes support 12-bit, while 16-bit medical scans require JPEG 2000, JPEG-LS, or uncompressed DICOM.

### Q50: How does your UI ensure that displayed images are not distorted by window resizing?
- **Direct Answer:** In `ui/sender/compression_view.py`, we implement a custom `ResizableImageLabel` class that overrides `resizeEvent` and scales the image pixmap using `Qt.AspectRatioMode.KeepAspectRatio` and `Qt.TransformationMode.SmoothTransformation`, preventing aspect ratio stretching or squashing.

---

# SECTION 5: SUMMARY CHECKLIST FOR MEMBER 1

Before stepping into the evaluation room, ensure you can:
- [ ] Draw the 5-stage JPEG compression pipeline on a whiteboard from memory.
- [ ] Write down the 2D-DCT formula and explain what $F(0, 0)$ is.
- [ ] Write down the formulas for MSE, PSNR, SSIM, CR, and Size Reduction %.
- [ ] Perform a manual 4-pixel MSE and PSNR calculation by hand in under 2 minutes.
- [ ] State the exact counts of the BUSI dataset: 780 total, 437 benign, 210 malignant, 133 normal; 546 train, 117 val, 117 test.
- [ ] Explain why nearest-neighbor interpolation is mandatory for binary masks.
- [ ] Explain why $8 \times 8$ blocking artifacts destroy lesion boundaries at Q30.
- [ ] Deliver your 5-minute presentation script fluently without reading from paper.
- [ ] Seamlessly hand over to Member 2 at the conclusion of your presentation.
