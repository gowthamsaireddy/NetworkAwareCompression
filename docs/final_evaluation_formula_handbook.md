# Mathematical and Technical Formula Handbook

**Project**: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation  
**Document Purpose**: Definitive technical reference detailing every mathematical equation, metric calculation, algorithmic derivation, variable definition, unit, numerical example, and source code location used across the project.

---

## 1. Summary Formula Quick Reference

| Metric / Parameter | Mathematical Formula | Units | Code Location | Key Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Compression Ratio (CR)** | $\text{CR} = \frac{S_{\text{orig}}}{S_{\text{comp}}}$ | Unitless (Ratio : 1) | `sender/compression.py:90`, `sender/region_adaptive_compressor.py:204` | Higher means smaller file relative to original |
| **Percentage Size Reduction** | $\Delta S\% = \left(\frac{S_{\text{orig}} - S_{\text{comp}}}{S_{\text{orig}}}\right) \times 100$ | Percentage (%) | `sender/compression.py:92`, `sender/region_adaptive_compressor.py:205` | Bandwidth saved during transmission |
| **Mean Squared Error (MSE)** | $\text{MSE} = \frac{1}{M \cdot N}\sum_{i=1}^M \sum_{j=1}^N [I(i,j) - K(i,j)]^2$ | $(\text{Intensity})^2$ | `sender/compression.py:75`, `sender/region_adaptive_compressor.py:199` | 0.0 means perfect pixel-for-pixel identity |
| **Peak Signal-to-Noise Ratio (PSNR)** | $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$ | Decibels (dB) | `cv2.PSNR()` in `sender/compression.py:80` | $>40\text{ dB}$ is excellent; $<30\text{ dB}$ exhibits visual degradation |
| **Preserved ROI PSNR** | $\text{PSNR}_{\text{ROI}} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}_{\text{ROI}}}\right)$ | Decibels (dB) | `sender/region_adaptive_compressor.py:200` | Evaluated strictly over lesion + safety margin |
| **Structural Similarity (SSIM)** | $\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | Unitless $[-1, 1]$ | `skimage.metrics.structural_similarity` | $\ge 0.95$ preserves perceptual tissue texture |
| **Gaussian Feathering ($\sigma$)** | $\sigma = \max\left(\frac{m}{3}, 1.0\right)$ | Pixels (px) | `sender/region_adaptive_compressor.py:117` | Controls softness of boundary alpha transition |
| **Dual-Rate Alpha Fusion** | $I_{\text{fused}}(x,y) = \alpha(x,y)I_{\text{ROI}} + [1-\alpha(x,y)]I_{\text{bg}}$ | uint8 $[0, 255]$ | `sender/region_adaptive_compressor.py:182` | Eliminates sharp DCT blocking at margin boundary |
| **Dice Similarity Coefficient** | $\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ | Unitless $[0.0, 1.0]$ | `ui/receiver/evaluation_view.py:114`, `models/train_unet.py:110` | Overlap agreement between predicted & ground truth |
| **Intersection over Union (IoU)** | $\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ | Unitless $[0.0, 1.0]$ | `ui/receiver/evaluation_view.py:117`, `models/train_unet.py:113` | Jaccard index; strictly lower than or equal to Dice |
| **Physical Wire Throughput** | $T_{\text{wire}} = \frac{B_{\text{wire}} \times 8}{t_{\text{wire}} \times 1000}$ | Kilobits per second (Kbps) | `sender/tcp_client.py:189`, `receiver/tcp_server.py:186` | Real measured data rate across socket |
| **Round-Trip Latency (RTT)** | $\text{RTT} = t_{\text{ack\_recv}} - t_{\text{send\_start}}$ | Seconds (s) / ms | `sender/tcp_client.py:228` | Total socket turnaround time including receiver ACK |
| **MGDA Pareto Optimal Weight** | $\alpha^* = \text{clip}\left(\frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1\right)$ | Unitless $[0.0, 1.0]$ | `models/train_unet.py:145` | Dynamically balances BCE and Soft Dice losses |

---

## 2. In-Depth Mathematical Formulations & Implementations

### 2.1 Compression Ratio (CR)

#### Mathematical Formula
$$\text{CR} = \frac{S_{\text{orig}}}{S_{\text{comp}}}$$

#### Variable Definitions
* $S_{\text{orig}}$: Uncompressed input image size in bytes. In our implementation, for a $777 \times 578$ 8-bit grayscale image, $S_{\text{orig}} = 777 \times 578 \times 1 = 449,106\text{ bytes}$ (or raw uncompressed disk file bytes e.g. PNG size).
* $S_{\text{comp}}$: Total byte volume of the compressed JPEG bitstream.
* **Units**: Dimensionless ratio, customarily expressed as $\text{CR}:1$.

#### Code Implementation
In `sender/compression.py` (lines 89–90) and `sender/region_adaptive_compressor.py` (lines 204):
```python
ratio = (orig_size / comp_size) if comp_size > 0 else 1.0
```

#### Step-by-Step Numerical Example
* Original BUSI image `benign (104)`: $S_{\text{orig}} = 456,327\text{ bytes}$.
* Compressed under Region-Adaptive Q90/Q20: $S_{\text{comp}} = 48,481\text{ bytes}$.
$$\text{CR} = \frac{456327}{48481} \approx 9.412\text{ (reported as 9.41:1)}$$

#### Interpretation
* $\text{CR} = 1.0$: No compression.
* $\text{CR} > 1.0$: Data compression achieved. Higher values indicate greater data reduction. In our project, Q30 achieves $\text{CR} \approx 18.56:1$, while Q90 achieves $\text{CR} \approx 6.07:1$.

---

### 2.2 Percentage Size Reduction ($\Delta S\%$)

#### Mathematical Formula
$$\Delta S\% = \left(\frac{S_{\text{orig}} - S_{\text{comp}}}{S_{\text{orig}}}\right) \times 100\% = \left(1 - \frac{1}{\text{CR}}\right) \times 100\%$$

#### Variable Definitions
* $S_{\text{orig}}$: Original image size in bytes.
* $S_{\text{comp}}$: Compressed bitstream size in bytes.
* **Units**: Percentage (%).

#### Code Implementation
In `sender/compression.py` (line 92):
```python
reduction_pct = max(0.0, ((orig_size - comp_size) / orig_size) * 100.0) if orig_size > 0 else 0.0
```

#### Step-by-Step Numerical Example
* $S_{\text{orig}} = 456,327\text{ B}$, $S_{\text{comp}} = 48,481\text{ B}$.
$$\Delta S\% = \left(\frac{456327 - 48481}{456327}\right) \times 100\% = \frac{407846}{456327} \times 100\% = 89.38\%$$

#### Interpretation
Measures network bandwidth conserved. An $89.38\%$ size reduction means the transmission link only needs to carry $10.62\%$ of the original data volume.

---

### 2.3 Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)

#### Mathematical Formula
$$\text{MSE} = \frac{1}{M \cdot N} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I_{\text{orig}}(i,j) - I_{\text{recon}}(i,j)]^2$$

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$

#### Variable Definitions
* $M, N$: Height and width of the ultrasound frame in pixels.
* $I_{\text{orig}}(i,j)$: Pixel intensity of original uncompressed image at row $i$, column $j$.
* $I_{\text{recon}}(i,j)$: Pixel intensity of reconstructed image after JPEG decompression.
* $\text{MAX}_I$: Maximum possible pixel intensity. For standard 8-bit grayscale medical ultrasound frames, $\text{MAX}_I = 2^8 - 1 = 255$.
* **Units**: $\text{MSE}$ is in $(\text{intensity levels})^2$; $\text{PSNR}$ is in Decibels ($\text{dB}$).

#### Code Implementation
In `sender/compression.py` (lines 75–80):
```python
mse = float(np.mean((gray.astype(np.float32) - decompressed.astype(np.float32)) ** 2))
psnr = float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-10)))
```

#### Step-by-Step Numerical Example
* Suppose $\text{MSE} = 4.25$ intensity units$^2$:
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{4.25}\right) = 10 \cdot \log_{10}\left(\frac{65025}{4.25}\right) = 10 \cdot \log_{10}(15300) = 10 \times 4.1847 = 41.85\text{ dB}$$

#### Interpretation
* Higher PSNR indicates higher fidelity to the original uncompressed image.
* $\text{PSNR} > 40\text{ dB}$: Excellent visual quality; differences are nearly imperceptible to human eye.
* $30\text{ dB} \le \text{PSNR} \le 40\text{ dB}$: Good quality; minor high-frequency compression softening.
* $\text{PSNR} < 30\text{ dB}$: Significant loss of diagnostic ultrasound speckle texture.
* **Clinical Warning**: PSNR is an overall mathematical pixel average. A high full-image PSNR does **not** guarantee that delicate lesion border margins were preserved!

---

### 2.4 Preserved Region-of-Interest PSNR ($\text{PSNR}_{\text{ROI}}$)

#### Mathematical Formula
$$\text{MSE}_{\text{ROI}} = \frac{1}{\sum_{(i,j)} \mathcal{M}_{\text{dilated}}(i,j)} \sum_{(i,j) \in \Omega_{\text{ROI}}} [I_{\text{orig}}(i,j) - I_{\text{recon}}(i,j)]^2$$

$$\text{PSNR}_{\text{ROI}} = 10 \cdot \log_{10}\left(\frac{255^2}{\max(\text{MSE}_{\text{ROI}}, 10^{-10})}\right)$$

#### Variable Definitions
* $\mathcal{M}_{\text{dilated}}$: Binary mask expanded by margin $m$ ($\mathcal{M}_{\text{dilated}}(i,j) \in \{0, 1\}$).
* $\Omega_{\text{ROI}} = \{(i,j) \mid \mathcal{M}_{\text{dilated}}(i,j) = 1\}$: The subset of pixels corresponding strictly to the lesion and surrounding safety margin.
* **Units**: Decibels ($\text{dB}$).

#### Code Implementation
In `sender/region_adaptive_compressor.py` (lines 197–203):
```python
roi_pixels = dilated_mask > 0
if np.any(roi_pixels):
    mse_roi = float(np.mean((orig_gray[roi_pixels].astype(np.float32) - dec_arr[roi_pixels].astype(np.float32)) ** 2))
    roi_psnr = float(10.0 * np.log10((255.0 ** 2) / max(mse_roi, 1e-10)))
else:
    roi_psnr = psnr_val
```

#### Step-by-Step Numerical Example
* For `benign (104)` with Option B (Region-Adaptive Q90 ROI / Q20 background):
  * Full Image PSNR: **36.42 dB** (pulled down by heavily compressed Q20 background tissue).
  * Preserved ROI PSNR: **49.70 dB** (computed strictly inside the diagnostic tumor region!).
* **Key Defense Point**: Proves that although the overall image was compressed heavily to save bandwidth, the diagnostic tumor region was preserved at near-lossless $49.70\text{ dB}$.

---

### 2.5 Structural Similarity Index Measure (SSIM)

#### Mathematical Formula
$$\text{SSIM}(x, y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$$
With standard weighting $\alpha = \beta = \gamma = 1$:
$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

#### Variable Definitions
* $\mu_x, \mu_y$: Local mean intensities of windows $x$ and $y$.
* $\sigma_x^2, \sigma_y^2$: Local variances of windows $x$ and $y$ (measuring contrast).
* $\sigma_{xy}$: Local covariance between windows $x$ and $y$ (measuring structural correlation).
* $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$: Small stabilizing constants where $L = 255$, $K_1 = 0.01$, $K_2 = 0.03$.
  * $C_1 = (0.01 \times 255)^2 = 2.55^2 = 6.5025$.
  * $C_2 = (0.03 \times 255)^2 = 7.65^2 = 58.5225$.
* **Units**: Dimensionless index bounded in $[-1.0, 1.0]$. In image compression, practically $[0.0, 1.0]$.

#### Code Implementation
In `sender/compression.py` (line 81) and `sender/region_adaptive_compressor.py` (line 194):
```python
ssim_val = float(ssim_fn(orig_gray, dec_arr, data_range=255))
```

#### Step-by-Step Numerical Example
* Local statistics: $\mu_x = 120.0$, $\mu_y = 118.5$, $\sigma_x^2 = 450.0$, $\sigma_y^2 = 440.0$, $\sigma_{xy} = 438.0$.
$$\text{Luminance Term} = \frac{2(120)(118.5) + 6.5}{120^2 + 118.5^2 + 6.5} = \frac{28440 + 6.5}{14400 + 14042.25 + 6.5} = \frac{28446.5}{28448.75} \approx 0.9999$$
$$\text{Contrast/Structure Term} = \frac{2(438) + 58.52}{450 + 440 + 58.52} = \frac{876 + 58.52}{890 + 58.52} = \frac{934.52}{948.52} \approx 0.9852$$
$$\text{SSIM} = 0.9999 \times 0.9852 = 0.9851$$

#### Interpretation
Captures human visual perception of acoustic shadows, boundaries, and hypoechoic tumor textures far better than MSE alone. $\text{SSIM} \ge 0.95$ indicates high structural fidelity.

---

### 2.6 Gaussian Alpha Feathering and Dual-Rate Fusion

#### Mathematical Formula
Let $\mathcal{M}_{\text{dilated}}(x,y) \in \{0, 255\}$ be the expanded lesion mask. The continuous feathering field $\alpha(x,y)$ is computed by convolving with a 2D Gaussian filter:
$$\alpha(x,y) = \text{clip}\left(\frac{\mathcal{M}_{\text{dilated}}(x,y) * G_\sigma(x,y)}{255.0}, 0.0, 1.0\right)$$
Where the Gaussian kernel standard deviation is configured proportionally to margin radius $m$:
$$\sigma = \max\left(\frac{m}{3.0}, 1.0\right), \quad K_{\text{size}} = (2m + 1) \mid 1$$
The fused composite ultrasound frame before bitstream transmission is:
$$I_{\text{fused}}(x,y) = \alpha(x,y) \cdot I_{\text{ROI}}(x,y) + [1 - \alpha(x,y)] \cdot I_{\text{bg}}(x,y)$$

#### Variable Definitions
* $m$: Lesion expansion margin in pixels (configurable $5\text{ px} \le m \le 50\text{ px}$; default $15\text{ px}$).
* $I_{\text{ROI}}(x,y)$: Ultrasound frame encoded at pristine quality $Q_{\text{ROI}}$ (default 90).
* $I_{\text{bg}}(x,y)$: Ultrasound frame encoded at aggressive quality $Q_{\text{bg}}$ (default 20).
* $\alpha(x,y)$: Blending coefficient field. $\alpha = 1.0$ inside lesion; $\alpha = 0.0$ in far background; smooth transition along boundary.
* **Units**: Intensities in $[0, 255]$ uint8; $\alpha$ in $[0.0, 1.0]$ float32.

#### Code Implementation
In `sender/region_adaptive_compressor.py` (lines 112–120, 182):
```python
sigma = max(margin_px / 3.0, 1.0)
ksize = (margin_px * 2 + 1) | 1
blurred = cv2.GaussianBlur(dilated_mask.astype(np.float32), (ksize, ksize), sigma)
alpha = np.clip(blurred / 255.0, 0.0, 1.0)
fused = (alpha_3d * roi_dec + (1.0 - alpha_3d) * bg_dec).astype(np.uint8).squeeze()
```

#### Step-by-Step Numerical Example
* At a boundary transition pixel: $\alpha = 0.65$.
* High-quality pixel $I_{\text{ROI}} = 142$; background pixel $I_{\text{bg}} = 110$.
$$I_{\text{fused}} = 0.65 \times 142 + (1 - 0.65) \times 110 = 92.3 + 38.5 = 130.8 \rightarrow 131\text{ uint8}$$
* **Why this matters**: Prevents sharp 8×8 block boundary step artifacts that fool downstream deep neural networks into predicting false-positive lesion edges!

---

### 2.7 Segmentation Accuracy: Dice Similarity Coefficient & IoU

#### Mathematical Formulations
$$\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

#### Relationship Between Dice and IoU
$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \quad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$
Notice that $\text{IoU} \le \text{Dice}$ always holds.

#### Variable Definitions
* $A$: Ground truth binary lesion mask ($\mathcal{M}_{\text{GT}} \in \{0, 1\}$).
* $B$: U-Net predicted binary lesion mask ($\mathcal{M}_{\text{pred}} \in \{0, 1\}$).
* $\text{TP}$ (True Positives): Pixels where both prediction and ground truth are 1.
* $\text{FP}$ (False Positives): Pixels where prediction is 1 but ground truth is 0 (over-segmentation).
* $\text{FN}$ (False Negatives): Pixels where ground truth is 1 but prediction is 0 (missed lesion area).
* **Empty Mask Convention**: If $|A| = 0$ (normal case with no lesion) and $|B| = 0$, $\text{Dice} = 1.0$, $\text{IoU} = 1.0$. If $|A| = 0$ and $|B| > 0$, $\text{Dice} = 0.0$, $\text{IoU} = 0.0$.
* **Units**: Dimensionless scores in $[0.0, 1.0]$.

#### Code Implementation
In `ui/receiver/evaluation_view.py` (lines 104–120) and `models/train_unet.py` (lines 108–118):
```python
intersection = np.logical_and(pred_bin, gt_bin).sum()
total_pixels = pred_bin.sum() + gt_bin.sum()
dice = (2.0 * intersection) / total_pixels if total_pixels > 0 else (1.0 if pred_bin.sum() == 0 else 0.0)
union = np.logical_or(pred_bin, gt_bin).sum()
iou = intersection / union if union > 0 else (1.0 if pred_bin.sum() == 0 else 0.0)
```

#### Step-by-Step Numerical Example
* Ground-truth mask has 1,200 lesion pixels.
* Model predicts 1,100 lesion pixels.
* Overlap ($\text{TP}$) is 980 pixels.
* $\text{FP} = 1100 - 980 = 120\text{ px}$.
* $\text{FN} = 1200 - 980 = 220\text{ px}$.
$$\text{Dice} = \frac{2 \times 980}{2(980) + 120 + 220} = \frac{1960}{1960 + 340} = \frac{1960}{2300} \approx 0.8522$$
$$\text{IoU} = \frac{980}{980 + 120 + 220} = \frac{980}{1320} \approx 0.7424$$
$$\text{Check via formula}: \text{IoU} = \frac{0.8522}{2 - 0.8522} = \frac{0.8522}{1.1478} \approx 0.7424\text{ (Matches perfectly!)}$$

---

### 2.8 Physical Network Wire Throughput & Round-Trip Latency (RTT)

#### Mathematical Formulations
$$T_{\text{wire}} = \frac{B_{\text{wire}} \times 8}{t_{\text{wire}} \times 1000}\text{ (in Kbps)}$$

$$\text{RTT} = t_{\text{ack\_received}} - t_{\text{wire\_start}}\text{ (in seconds)}$$

#### Variable Definitions
* $B_{\text{wire}}$: Total physical byte count transmitted over the socket on the wire:
  $$B_{\text{wire}} = \text{Header (14 B)} + \text{len}(\text{Metadata JSON}) + \text{len}(\text{Image Bytes})$$
* $t_{\text{wire}}$: Elapsed physical time from first byte sent to final payload byte acknowledged by TCP kernel buffer:
  $$t_{\text{wire}} = \max(t_{\text{wire\_end}} - t_{\text{wire\_start}}, 10^{-6})\text{ s}$$
* **Units**: $B_{\text{wire}}$ in bytes; $t_{\text{wire}}$ in seconds; Throughput in Kilobits per second ($\text{Kbps}$).

#### Code Implementation
In `sender/tcp_client.py` (lines 173–189):
```python
total_bytes = IMAGE_HEADER_SIZE + len(metadata.to_json().encode("utf-8")) + len(image_bytes)
t_wire_start = time.perf_counter()
send_image_frame(self.sock, metadata, image_bytes, progress_callback=progress_callback, timeout=self.timeout)
t_wire_end = time.perf_counter()
wire_time_sec = max(t_wire_end - t_wire_start, 1e-6)
wire_throughput_kbps = (total_bytes * 8.0) / (wire_time_sec * 1000.0)
```
And on the Windows receiver in `receiver/tcp_server.py` (lines 185–186):
```python
total_wire_bytes = IMAGE_HEADER_SIZE + len(meta_json_bytes) + len(image_bytes)
measured_throughput_kbps = (total_wire_bytes * 8.0) / (rx_time_sec * 1000.0)
```

#### Step-by-Step Numerical Example
* Total wire bytes: $B_{\text{wire}} = 49,007\text{ bytes}$ (Image payload: 48,481 B + JSON metadata: 512 B + Header: 14 B).
* Measured transmission clock diff: $t_{\text{wire}} = 0.004547\text{ seconds}$ ($4.55\text{ ms}$).
$$\text{Bits Transmitted} = 49007 \times 8 = 392,056\text{ bits}$$
$$T_{\text{wire}} = \frac{392056}{0.004547 \times 1000} = \frac{392056}{4.547} \approx 86,223\text{ Kbps } (\approx 86.22\text{ Mbps})$$

#### Theoretical Bandwidth vs. Measured Throughput Distinction
* **Theoretical Bandwidth**: The configured physical capacity ceiling of a link (e.g. 1000 Mbps Gigabit Ethernet, or a 2 Mbps `tc/netem` shaped profile).
* **Measured Throughput**: The actual rate of successfully delivered application data bits over physical wall-clock time. It is always strictly lower than theoretical bandwidth due to TCP header overhead, operating system socket buffer context switching, and network latency.

---

### 2.9 Multi-Objective Loss and MGDA Pareto-Optimal Solver

#### Mathematical Formulations
In deep lesion segmentation, the network optimizes two conflicting objectives:
1. Binary Cross-Entropy Loss (measures pixel-wise calibration):
   $$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
2. Soft Dice Loss (measures region overlap directly):
   $$\mathcal{L}_{\text{Dice}}(y, \hat{y}) = 1 - \frac{2 \sum_{i=1}^N y_i \hat{y}_i + \epsilon}{\sum_{i=1}^N y_i + \sum_{i=1}^N \hat{y}_i + \epsilon}$$

In standard training, these are summed with arbitrary heuristic weights: $\mathcal{L} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$. However, their gradients $\mathbf{g}_1 = \nabla_{\theta} \mathcal{L}_{\text{BCE}}$ and $\mathbf{g}_2 = \nabla_{\theta} \mathcal{L}_{\text{Dice}}$ frequently point in conflicting directions ($\mathbf{g}_1^T \mathbf{g}_2 < 0$), causing gradient cancellation and plateauing.

The Multiple-Gradient Descent Algorithm (MGDA) formulates a Pareto-optimal quadratic optimization problem at every step:
$$\min_{\alpha \in [0, 1]} \|\alpha \mathbf{g}_1 + (1 - \alpha) \mathbf{g}_2\|_2^2$$
Expanding the norm squared:
$$\min_{\alpha \in [0, 1]} \alpha^2 \|\mathbf{g}_1\|^2 + 2\alpha(1-\alpha)\mathbf{g}_1^T \mathbf{g}_2 + (1-\alpha)^2 \|\mathbf{g}_2\|^2$$
Differentiating with respect to $\alpha$ and setting to zero yields the unconstrained minimizer $\hat{\alpha}$:
$$\hat{\alpha} = \frac{\|\mathbf{g}_2\|^2 - \mathbf{g}_1^T \mathbf{g}_2}{\|\mathbf{g}_1\|^2 + \|\mathbf{g}_2\|^2 - 2\mathbf{g}_1^T \mathbf{g}_2} = \frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}$$
Projecting onto the simplex constraint $[0, 1]$:
$$\alpha^* = \text{clip}(\hat{\alpha}, 0.0, 1.0)$$

#### Code Implementation
In `models/train_unet.py` (lines 142–155):
```python
diff = g2 - g1
denom = float(torch.dot(diff, diff))
if denom < 1e-12:
    alpha = 0.5
else:
    numerator = float(torch.dot(diff, g2))
    alpha = min(max(numerator / denom, 0.0), 1.0)
```

#### Step-by-Step Numerical Example
* Suppose at training step $k$:
  * $\|\mathbf{g}_1\|^2 = 4.0$ (BCE gradient norm squared)
  * $\|\mathbf{g}_2\|^2 = 9.0$ (Dice gradient norm squared)
  * $\mathbf{g}_1^T \mathbf{g}_2 = -2.0$ (Negative dot product $\rightarrow$ gradients conflict!)
* Compute:
$$\|\mathbf{g}_1 - \mathbf{g}_2\|^2 = \|\mathbf{g}_1\|^2 + \|\mathbf{g}_2\|^2 - 2\mathbf{g}_1^T \mathbf{g}_2 = 4.0 + 9.0 - 2(-2.0) = 13.0 + 4.0 = 17.0$$
$$\text{Numerator} = \|\mathbf{g}_2\|^2 - \mathbf{g}_1^T \mathbf{g}_2 = 9.0 - (-2.0) = 11.0$$
$$\alpha^* = \frac{11.0}{17.0} \approx 0.647$$
* Pareto-optimal weight: $\alpha^* = 0.647$ for BCE, and $1 - \alpha^* = 0.353$ for Dice.
* The combined gradient $\mathbf{g}^* = 0.647 \mathbf{g}_1 + 0.353 \mathbf{g}_2$ minimizes gradient conflict and guarantees non-negative descent on both objectives simultaneously!
* **Resulting Gain**: Boosted validation Dice from **0.6056** to **0.6581** (+5.25% absolute gain).

---

### 2.10 SHA-256 Cryptographic Hash Digest

#### Mathematical / Algorithmic Concept
SHA-256 is an iterative cryptographic hash function operating on 512-bit message blocks with a 256-bit internal state:
1. **Padding**: Appends bit `1`, followed by $k$ zero bits, and finally the 64-bit integer representation of message length $L$, such that the total length is a multiple of 512 bits: $(L + 1 + k) \equiv 448 \pmod{512}$.
2. **Compression Function**: Employs 64 rounds of non-linear logical operations (bitwise Ch, Maj, $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$) and 64 round constants derived from the cube roots of the first 64 prime numbers.
3. **Avalanche Effect**: Altering even a single bit in the 48,481-byte image payload alters on average $50\%$ of the output digest bits ($128$ bits).

#### Code Implementation
In `shared/protocol.py` (lines 35–37) and `receiver/integrity.py` (lines 14–16):
```python
def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```

#### Verification Example
* Image payload $I_1$ digest: `e8045fdacc8dc53fd7d04c79198e107374466feb45202cd51b01948949180609`.
* Tampered payload $I_2$ (1 bit flipped at byte 4000): `dfd61118b0fb73b8e45bc7bca72cbab50c6d5f9db891298b82a82f2b07240ef3`.
* Direct equality test `actual == expected` fails in **0.02 ms**, immediately halting pipeline before damaged data reaches U-Net.
