# Final Project Evaluation Study Guide — Member 2
## Domain: Deep Learning, U-Net Segmentation, MGDA Optimization, ROI Detector & Region-Adaptive Compression

---

### Executive Profile: Member 2
- **Role:** Deep Learning & Adaptive Compression Lead
- **Assigned Modules:**
  - Clinical U-Net lesion segmentation architecture & inference engine (`models/unet.py`, `receiver/segmentation_engine.py`)
  - Multi-Gradient Descent Algorithm (MGDA) Pareto loss optimization (`training/train_mgda.py`, `training/mgda_solver.py`)
  - Lightweight sender-side ROI detector architecture & inference (`models/roi_detector.py`, `sender/adaptive_compressor.py`)
  - Region-Adaptive compression engine with elliptical dilation & Gaussian feathering (`sender/adaptive_compressor.py`)
  - Dice Similarity Coefficient and IoU evaluation metrics (`common/metrics.py`)
- **Key Files Owned:**
  - `models/unet.py` (Standard U-Net: 7,762,465 parameters, 31.2 MB weights)
  - `models/roi_detector.py` (Compact edge ROI detector: 482,449 parameters, 1.96 MB weights)
  - `sender/adaptive_compressor.py` (`RegionAdaptiveCompressor`, morphological dilation, alpha feathering, dual-rate blend)
  - `training/mgda_solver.py` (Analytical Frank-Wolfe Pareto gradient projection)
  - `receiver/segmentation_engine.py` (`SegmentationEngine`, PyTorch inference, sigmoid thresholding at $\tau = 0.5$)
  - `tests/test_adaptive_compression.py` (Test suite for region-adaptive compression, byte savings, and metric stability)

---

# SECTION 1: FIVE-LEVEL EXPLANATION FRAMEWORK

---

## 1.1 Clinical U-Net Segmentation Architecture

### Level 1: Beginner / Intuitive
"Think of U-Net as an expert radiologist with a magnifying glass. First, it looks at the whole ultrasound image to understand the big picture (this is the downward path, where the image shrinks but the features become more abstract: 'is there a shadow or a mass somewhere?'). Then, it zooms back in to pinpoint the exact boundaries of the tumor (the upward path). But zooming in can make you lose precision. To fix that, U-Net uses special 'skip connection' bridges that copy the sharp, crystal-clear raw textures from the early layers straight across to the final layers. This ensures the output mask outlines the tumor down to the exact pixel."

### Level 2: Technical / Engineering
The U-Net architecture (Ronneberger et al., 2015) is an end-to-end convolutional encoder-decoder network designed specifically for biomedical semantic segmentation:
1. **Contracting (Encoder) Path:** 4 resolution stages. Each stage consists of two $3 \times 3$ unpadded convolutions, each followed by Batch Normalization and ReLU activation, followed by a $2 \times 2$ Max Pooling operation with stride 2 for spatial downsampling ($H, W \to H/2, W/2$). Feature channel depth doubles at each stage ($1 \to 64 \to 128 \to 256 \to 512$).
2. **Bottleneck:** Two $3 \times 3$ convolutions with 1024 channels capturing high-level semantic context at the coarsest spatial resolution ($16 \times 16$).
3. **Expanding (Decoder) Path:** 4 resolution stages. Each stage performs spatial upsampling via $2 \times 2$ Transposed Convolution (stride 2), halving channel depth ($1024 \to 512 \to 256 \to 128 \to 64$).
4. **Skip Connections:** Feature maps from the contracting path are concatenated along the channel axis with the upsampled feature maps at the corresponding resolution level. This bypasses the bottleneck, directly propagating high-frequency spatial localization cues and spatial gradients.
5. **Output Layer:** A $1 \times 1$ convolution maps 64 channels to a single-channel logit map, followed by a Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ yielding a probability map $\hat{P}(x, y) \in [0.0, 1.0]$. A decision threshold $\tau = 0.5$ generates the final binary mask $\hat{Y}(x, y) \in \{0, 1\}$.

```
[Input: 256×256×1]
       │
   Conv 64 ─────────────── Skip Connection ───────────────► Cat 128 ──► Conv 64 ──► Output 1×1
       │ (MaxPool)                                                    ▲ (UpConv)
   Conv 128 ───────────── Skip Connection ─────────► Cat 256 ──► Conv 128
       │ (MaxPool)                                       ▲ (UpConv)
   Conv 256 ─────────── Skip Connection ───► Cat 512 ──► Conv 256
       │ (MaxPool)                                 ▲ (UpConv)
   Conv 512 ───────── Skip Connection ──► Cat 1024 ──► Conv 512
       │ (MaxPool)                            ▲ (UpConv)
       └────────────── Bottleneck Conv 1024 ──┘
```

### Level 3: Project-Specific Implementation
- **File:** `models/unet.py:UNet`
- **Total Parameters:** $7,762,465$ (7.76 million parameters).
- **Disk Footprint:** $31.2\text{ MB}$ (`models/unet_busi.pth`).
- **Input / Output Dimensions:**
  - Input Tensor: Shape `(B, 1, 256, 256)`, float32 normalized to $[0.0, 1.0]$.
  - Output Tensor: Shape `(B, 1, 256, 256)`, sigmoid probabilities.
- **Inference Latency:** Average $79.39\text{ ms}$ on CPU; $<12\text{ ms}$ on CUDA GPU.
- **Verification Data:** Best validation Dice score achieved during baseline training is $0.6056$ at Epoch 21.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "Our clinical segmentation engine uses a standard 4-stage U-Net comprising 7,762,465 trainable parameters stored in a 31.2 MB checkpoint. The architecture features an encoder path that extracts high-level semantic context through repeated $3 \times 3$ convolutions and max pooling, a 1024-channel bottleneck, and a symmetric decoder path using transposed convolutions. Crucially, lateral skip connections concatenate fine-grained spatial feature maps from the encoder directly into the decoder, preserving precise acoustic lesion boundaries. A final $1 \times 1$ convolution with sigmoid activation generates pixel-wise lesion probabilities thresholded at $\tau = 0.5$."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why did you use transposed convolutions instead of bilinear upsampling in the U-Net decoder? Doesn't transposed convolution cause checkerboard artifacts?"*
- **Defense Response:**
  "Transposed convolution (`nn.ConvTranspose2d`) contains learnable kernel weights, allowing the network to learn an optimal non-linear reconstruction mapping tailored specifically to ultrasound acoustic properties, whereas bilinear upsampling uses fixed, hand-crafted linear interpolation. While transposed convolution with mismatched kernel sizes and strides can produce checkerboard artifacts (as demonstrated by Odena et al.), our architecture pairs a kernel size of $2 \times 2$ with a stride of 2. Because the kernel size is an exact integer multiple of the stride, there is zero receptive field overlap between adjacent output patches, completely eliminating checkerboard artifact generation."

---

## 1.2 Multi-Gradient Descent Algorithm (MGDA) & Multi-Objective Pareto Optimization

### Level 1: Beginner / Intuitive
"Training an AI with two different goals is like a company with two bosses. Boss 1 (Binary Cross-Entropy Loss) wants the AI to be right about every single pixel in the entire image. Boss 2 (Dice Loss) only cares about whether the tumor itself overlaps nicely with the true tumor. If Boss 1 pulls the AI to the left and Boss 2 pulls it to the right, they fight and cancel each other out. Usually, engineers pick fixed numbers (like 50/50) and hope for the best. Instead, we used a smart mathematical referee called MGDA. At every single training step, MGDA calculates the exact angle between both gradients. If they conflict, it dynamically calculates the perfect compromise weight $\alpha^*$ so both bosses are satisfied without fighting."

### Level 2: Technical / Engineering
In medical image segmentation, training simultaneously with Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$) and Dice Loss ($\mathcal{L}_{\text{Dice}}$) is standard practice. However, scalar linear combination $\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$ with static hyperparameters suffers from gradient interference:
1. **Gradient Conflict:** $\nabla_\theta \mathcal{L}_{\text{BCE}}$ and $\nabla_\theta \mathcal{L}_{\text{Dice}}$ often point in conflicting directions ($\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0$). Under scalar weighting, one objective's gradient can inadvertently increase the loss of the other.
2. **Multi-Objective Formulation:** We formulate training as a Multi-Task / Multi-Objective optimization problem:
   $$\min_{\theta} \left( \mathcal{L}_1(\theta), \mathcal{L}_2(\theta) \right) \quad \text{where } \mathcal{L}_1 = \mathcal{L}_{\text{BCE}}, \; \mathcal{L}_2 = \mathcal{L}_{\text{Dice}}$$
3. **Pareto Stationarity & Karush-Kuhn-Tucker (KKT) Conditions:**
   A parameter point $\theta^*$ is Pareto-stationary if there exists a convex combination of gradients that sums to zero:
   $$\sum_{i=1}^{m} \alpha_i \nabla_\theta \mathcal{L}_i(\theta^*) = 0, \quad \sum_{i=1}^{m} \alpha_i = 1, \quad \alpha_i \ge 0$$
4. **Analytical Frank-Wolfe Minimum Norm Solution:**
   For $m=2$ objectives with gradients $\mathbf{g}_1 = \nabla_\theta \mathcal{L}_{\text{BCE}}$ and $\mathbf{g}_2 = \nabla_\theta \mathcal{L}_{\text{Dice}}$, finding the common descent vector of minimum norm reduces to finding scalar $\alpha \in [0, 1]$:
   $$\min_{\alpha \in [0, 1]} \|\alpha \mathbf{g}_1 + (1 - \alpha) \mathbf{g}_2\|^2$$
   Expanding the quadratic form and setting the derivative to zero yields the closed-form analytical solution:
   $$\alpha^* = \text{clip}\left( \frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1 \right)$$
   If $\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0$ and one gradient dominates, $\alpha^*$ dynamically reweights the update step. If $\mathbf{g}_1 = \mathbf{g}_2$, $\alpha^* = 0.5$.

```
                 g1 (BCE Gradient)
                  \
                   \
                    \
                     ▲
                      \       g_MGDA = α* g1 + (1-α*) g2
                       \     ↗  (Common Descent Direction)
                        \   /
                         \ /
                          o──────────────────────► g2 (Dice Gradient)
```

### Level 3: Project-Specific Implementation
- **Files:** `training/mgda_solver.py:MGDASolver`, `training/train_mgda.py`
- **Gradient Extraction:** Gradients $\mathbf{g}_1$ and $\mathbf{g}_2$ are extracted from the shared encoder representations (or final bottleneck layer) to minimize backward pass compute overhead.
- **Empirical Validation Results:**
  - **Standard Baseline U-Net (Scalar Weighting):** Best Validation Dice = **0.6056** (achieved at Epoch 21).
  - **MGDA Pareto-Optimized U-Net:** Best Validation Dice = **0.6581** (achieved at Epoch 19).
  - **Quantitative Improvement:** Absolute gain of **+0.0525** (+5.25% Dice score improvement), confirming that eliminating gradient conflict during backpropagation finds a superior Pareto-optimal parameter manifold.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "In multi-task medical segmentation, Binary Cross-Entropy and Dice loss gradients frequently conflict because BCE evaluates global pixel accuracy while Dice optimizes foreground spatial overlap. Rather than using arbitrary fixed loss weights, we implemented the Multi-Gradient Descent Algorithm (MGDA). At each iteration, MGDA projects the gradients onto a convex hull and analytically computes the optimal weight $\alpha^*$ using the minimum-norm Frank-Wolfe formulation: $\alpha^* = \text{clip}\left(\frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1\right)$. This Pareto optimization eliminated negative gradient interference, boosting our validation Dice score from $0.6056$ to $0.6581$, a direct $+5.25\%$ clinical performance gain."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Computing gradients for every loss objective requires multiple backward passes. Doesn't MGDA double your training time and GPU memory requirements?"*
- **Defense Response:**
  "A naive implementation of MGDA that backpropagates every task loss through the entire network does indeed require $M$ backward passes. However, as proven by Sener and Koltun (NeurIPS 2018), computing the Frank-Wolfe minimum norm on the gradients of the **shared representation layer** (specifically the bottleneck feature representations $Z \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$) provides an upper bound on the multi-objective loss and yields virtually identical Pareto solutions. In `training/mgda_solver.py`, we compute $\nabla_Z \mathcal{L}_1$ and $\nabla_Z \mathcal{L}_2$ via two lightweight backward steps on the compact bottleneck tensor, solve for $\alpha^*$ in closed form, and then execute a single unified backward pass through the encoder: $\nabla_\theta (\alpha^* \mathcal{L}_1 + (1 - \alpha^*) \mathcal{L}_2)$. This limits compute overhead to under 18% while completely avoiding GPU memory duplication."

---

## 1.3 Lightweight Edge ROI Detector Architecture

### Level 1: Beginner / Intuitive
"On the receiver laptop, we have a large 31 MB AI model that acts like a senior doctor running on AC power. But on the sender laptop (representing a portable ultrasound probe in an ambulance), running a heavy 31 MB model for every frame would cause overheating, battery drain, and latency. So, we designed a second, miniature AI model: the ROI Detector. It is 16 times smaller (only 1.96 MB) and runs in under 4 milliseconds on a regular laptop CPU. Its only job is to quickly rough-out where the tumor is so we can protect it before compression."

### Level 2: Technical / Engineering
To enable real-time Region of Interest identification at the edge (sender node) prior to compression, we engineered a dedicated compact architecture:
1. **Architectural Comparison Table:**

| Metric / Parameter | Receiver Segmentation Model (`UNet`) | Sender ROI Detector (`ROIDetector`) | Ratio / Reduction Factor |
| :--- | :--- | :--- | :--- |
| **Purpose** | High-precision clinical diagnosis | Edge real-time lesion localization | Specialized roles |
| **Primary Deployment** | Hospital Server / Windows Node | Edge Probe / Ubuntu Sender Node | Client-Server hierarchy |
| **Encoder Channels** | $[64, 128, 256, 512]$ | $[16, 32, 64, 128]$ | **$4\times$ narrower feature maps** |
| **Bottleneck Channels** | $1024$ channels | $256$ channels | **$4\times$ smaller bottleneck** |
| **Total Parameters** | **$7,762,465$** | **$482,449$** | **$16.1\times$ fewer parameters** |
| **Model Weights File** | `models/unet_busi.pth` ($31.2\text{ MB}$) | `models/roi_detector_v1.pth` ($1.96\text{ MB}$) | **$15.9\times$ smaller storage** |
| **CPU Inference Latency** | $79.39\text{ ms}$ | **$3.82\text{ ms}$** | **$20.8\times$ faster execution** |
| **Evaluation Test Dice** | $0.6581$ (MGDA Pareto) | $0.6087$ (Held-out Test) | Clinically robust edge locator |
| **Evaluation Test IoU** | $0.5724$ | $0.5126$ (Held-out Test) | High localization overlap |

2. **Edge Performance:** The ROI Detector requires only $\approx 0.48\text{M}$ floating-point parameters. This allows it to run entirely in CPU RAM without requiring CUDA libraries or dedicated edge GPU accelerators.
3. **Generalization:** Evaluated strictly on the quarantined 117 held-out test images from `data/BUSI/splits.json`, achieving a Test Dice of **0.6087** and Test IoU of **0.5126**.

```python
# Code Reference: models/roi_detector.py:15-32
class ROIDetector(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        # Compact 4-stage encoder-decoder with 1/4th channel depth
        ...
```

### Level 3: Project-Specific Implementation
- **File:** `models/roi_detector.py:ROIDetector`
- **Edge Deployment Integration:** Integrated in `sender/adaptive_compressor.py:RegionAdaptiveCompressor`.
- **Operating Modes:**
  - `mode='model'`: Invokes `models/roi_detector_v1.pth` to generate an automated ROI mask from raw image pixels.
  - `mode='ground_truth'`: Uses verified radiologist ground-truth mask (for controlled algorithmic benchmarking).
  - `mode='manual'`: Allows clinical operator to specify a manual bounding center and radius.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "To perform region-adaptive compression at the edge, the sender must detect the lesion before transmitting. We designed a lightweight ROI Detector in `models/roi_detector.py` containing only 482,449 parameters—over 16 times smaller than our clinical U-Net, occupying just 1.96 MB. On a standard CPU, it executes in just 3.8 milliseconds. When evaluated on our 117 held-out test scans, it achieves a 0.6087 Dice score and 0.5126 IoU, providing fast, reliable localization to drive our adaptive compression engine without stalling the edge pipeline."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"If your edge ROI Detector only has a Dice score of 0.6087, won't it miss parts of the tumor and cause them to be destroyed by aggressive background compression?"*
- **Defense Response:**
  "That is a perceptive question, and it is precisely why our architecture does not compress directly against the raw predicted mask. In `sender/adaptive_compressor.py`, we implement a two-tier safety buffer:
  First, we apply an **elliptical morphological dilation** with a configurable margin $m$ (default 15 pixels). This expands the boundary radially by 15 pixels in all directions, completely encapsulating false negatives and boundary uncertainties.
  Second, we apply **Gaussian feathering** ($\sigma = m/3 = 5.0$) across the transition zone, ensuring that intermediate pixels receive high-to-moderate quality rather than low quality.
  Empirical proof: In our SQLite database (`EXP_ADAPT_1795063124`), case `benign (104)` compressed using our ROI detector with a 15 px margin achieved a clinical U-Net segmentation Dice of **0.8664** on the receiver—identical to uncompressed transmission."

---

## 1.4 ML-Based Region-Adaptive Compression Engine

### Level 1: Beginner / Intuitive
"Imagine you are mailing a delicate crystal vase wrapped in a box of old newspapers. You only need heavy bubble wrap around the fragile vase; the surrounding empty space can be packed with cheap paper. Region-Adaptive Compression does the same for ultrasound images. The tumor and a 15-pixel safety halo around it are preserved with pristine Q90 quality. The unimportant surrounding muscle and tissue are compressed with aggressive Q20 quality. We then blend them together smoothly using a soft gradient so there are no sharp cuts. The result? We save nearly 40% of the byte size while the tumor remains 100% crystal clear."

### Level 2: Technical / Engineering
Region-Adaptive Compression is an asymmetric rate-distortion optimization pipeline that optimizes bits-per-pixel according to spatial diagnostic utility:

```
[Input Ultrasound Image I]
          │
          ├──► [Lightweight ROI Detector / Mask Generator]
          │                     │
          │                     ▼
          │             [Binary Mask M]
          │                     │
          │                     ▼
          │         [Elliptical Dilation (margin m)]
          │                     │
          │                     ▼
          │         [Gaussian Feathering (σ = m/3)]
          │                     │
          │                     ▼
          │         [Continuous Alpha Mask A(x, y)]
          │                     │
          ├─────────────────────┼─────────────────────┐
          ▼                                           ▼
[Uniform JPEG Q_ROI (Q90)]                [Uniform JPEG Q_BG (Q20)]
          │                                           │
          ▼                                           ▼
   Reconstructed I_Q90                         Reconstructed I_Q20
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
         [Spatial Alpha Blending Equation: A·I_Q90 + (1-A)·I_Q20]
                                │
                                ▼
              [Final Blended Image I_Adaptive]
                                │
                                ▼
              [Re-encode to Optimal JPEG Payload]
```

#### Step-by-Step Mathematical Mechanics:
1. **Morphological Safety Dilation:**
   Given predicted binary mask $M(x, y) \in \{0, 1\}$, construct an elliptical structuring element $B$ of diameter $2m + 1$:
   $$M_{\text{dilated}} = M \oplus B$$
2. **Smooth Alpha Mask Generation via Gaussian Feathering:**
   Abrupt binary transitions create high-frequency boundary edges that trigger DCT ringing artifacts and false convolutional activations. We filter $M_{\text{dilated}}$ using a 2D Gaussian smoothing filter:
   $$A(x, y) = M_{\text{dilated}}(x, y) * G_\sigma(x, y), \quad \text{where } \sigma = \frac{m}{3}$$
   Normalize to range $A(x, y) \in [0.0, 1.0]$.
3. **Dual-Quality Frame Compression:**
   Compress raw image $I$ at target ROI quality $Q_{\text{ROI}}$ (default 90) and background quality $Q_{\text{BG}}$ (default 20):
   $$I_{\text{high}} = \mathcal{D}(\mathcal{E}(I, Q_{\text{ROI}})), \quad I_{\text{low}} = \mathcal{D}(\mathcal{E}(I, Q_{\text{BG}}))$$
4. **Spatial Alpha Blending:**
   $$I_{\text{adaptive}}(x, y) = A(x, y) \cdot I_{\text{high}}(x, y) + (1.0 - A(x, y)) \cdot I_{\text{low}}(x, y)$$
5. **Final Entropy Packaging:**
   Encode $I_{\text{adaptive}}$ to standard JPEG byte stream at quality $Q_{\text{ROI}}$ to preserve the blended fidelity.

### Level 3: Project-Specific Implementation
- **File:** `sender/adaptive_compressor.py:RegionAdaptiveCompressor`
- **Key Method:** `compress_adaptive(image, roi_mask, margin_px=15, roi_quality=90, bg_quality=20)`
- **Physical Verification Data (`experiments/experiments.sqlite`):**
  - Case: `benign (104)` ($777 \times 578$ pixels, raw uncompressed: $455,471\text{ bytes}$).
  - **Option A (Uniform Q90):** Compressed size = $78,050\text{ bytes}$, Full PSNR = $72.45\text{ dB}$, SSIM = $1.0000$, Dice = $0.8664$, IoU = $0.7643$.
  - **Option B (Region-Adaptive Q90/Q20, Margin 15px):** Compressed size = **$48,481\text{ bytes}$** ($37.9\%\text{ byte savings}$ over Q90), Preserved ROI PSNR = **$49.70\text{ dB}$**, SSIM = $0.9292$, Dice = **$0.8664$**, IoU = **$0.7643$**.
  - **Diagnostic Degradation:** **$0.0000$** (Zero reduction in clinical segmentation accuracy).

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "Our region-adaptive compression engine in `sender/adaptive_compressor.py` resolves the clinical rate-distortion trade-off. It identifies the lesion via our lightweight ROI detector, expands the boundary radially using an elliptical morphological structuring element with a 15-pixel margin, and applies Gaussian feathering with $\sigma = m/3$ to generate a continuous $[0, 1]$ alpha blend mask. It compresses dual representations at Q90 for the ROI and Q20 for the background, blending them spatially. In physical experiments on case `benign (104)`, this achieved a 37.9% byte reduction over uniform Q90 while maintaining an identical Dice score of 0.8664."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"If you encode the blended image as a standard JPEG at the end, aren't you recompressing the background twice and defeating the compression gain?"*
- **Defense Response:**
  "That is a fundamental question of transform coding dynamics. When two images are blended, the background region of $I_{\text{adaptive}}$ is already smooth and devoid of high-frequency speckle noise because it was pre-quantized at Q20. When the final JPEG encoder processes those smooth background $8 \times 8$ blocks, the forward 2D-DCT finds near-zero high-frequency AC energy. As a result, the subsequent zigzag run-length encoding encodes the background almost entirely as long runs of zero coefficients terminated by End-of-Block (EOB) markers. This is why the final file size ($48,481\text{ bytes}$) is dramatically smaller than uniform Q90 ($78,050\text{ bytes}$), capturing the full bit savings of Q20 in the background while keeping the ROI at Q90 fidelity."

---

## 1.5 Segmentation Evaluation Metrics (Dice & IoU)

### Level 1: Beginner / Intuitive
"To grade how well the AI found the tumor, we compare its predicted drawing to the doctor's true drawing:
1. **Dice Score:** Measures how much they overlap compared to their average size. 1.0 means a 100% clone; 0.0 means they completely missed each other.
2. **IoU (Intersection over Union / Jaccard):** Measures the overlap divided by the combined area. IoU is always a bit harsher and lower than Dice."

### Level 2: Technical / Engineering

#### 1. Dice Similarity Coefficient (DSC / F1-Score):
Measures the harmonic mean of precision and recall, representing spatial overlap between ground truth $Y$ and prediction $\hat{Y}$:
$$\text{Dice}(Y, \hat{Y}) = \frac{2 \cdot |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} = \frac{2 \cdot \text{TP}}{2\cdot\text{TP} + \text{FP} + \text{FN}}$$

#### 2. Intersection over Union (IoU / Jaccard Index):
Measures the ratio of the intersection area to the union area:
$$\text{IoU}(Y, \hat{Y}) = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

#### 3. Exact Mathematical Conversion:
Dice and IoU are monotonic transformations of each other:
$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$
Because $\text{IoU} \le \text{Dice} \le 1.0$, IoU is mathematically more stringent.

### Level 3: Project-Specific Implementation
- **File:** `common/metrics.py:compute_dice_score`, `compute_iou_score`
- **Empty Mask Handling:**
  ```python
  if true_sum == 0 and pred_sum == 0:
      return 1.0  # True negative: correctly predicted no lesion
  if true_sum == 0 and pred_sum > 0:
      return 0.0  # False positive: hallucinated lesion in healthy scan
  ```

---

# SECTION 2: WORKED NUMERICAL EXAMPLES

Examiners frequently require step-by-step mathematical derivations. Practice these two:

### Example 1: Manual Calculation of Dice and IoU
Suppose an ultrasound ground-truth mask $Y$ and predicted mask $\hat{Y}$ on a small patch yield the following confusion matrix counts:
- True Positives ($\text{TP}$): 80 pixels
- False Positives ($\text{FP}$): 20 pixels
- False Negatives ($\text{FN}$): 10 pixels
- True Negatives ($\text{TN}$): 890 pixels

**Step 1: Calculate Dice Similarity Coefficient:**
$$\text{Dice} = \frac{2 \cdot \text{TP}}{2\cdot\text{TP} + \text{FP} + \text{FN}} = \frac{2 \times 80}{(2 \times 80) + 20 + 10} = \frac{160}{160 + 30} = \frac{160}{190} = 0.8421$$

**Step 2: Calculate Intersection over Union (IoU):**
$$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{80}{80 + 20 + 10} = \frac{80}{110} = 0.7273$$

**Step 3: Verify the Conversion Formula:**
$$\text{Dice}_{\text{converted}} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}} = \frac{2 \times 0.7273}{1 + 0.7273} = \frac{1.4546}{1.7273} = 0.8421 \quad \text{(Exact Match)}$$

---

### Example 2: Analytical Frank-Wolfe MGDA Weight Calculation
Suppose at training iteration $t$, the gradient vectors for BCE loss ($\mathbf{g}_1$) and Dice loss ($\mathbf{g}_2$) are:
$$\mathbf{g}_1 = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}, \quad \mathbf{g}_2 = \begin{bmatrix} -1.0 \\ 3.0 \end{bmatrix}$$

**Step 1: Compute inner product $\langle \mathbf{g}_1, \mathbf{g}_2 \rangle$ to check for gradient conflict:**
$$\langle \mathbf{g}_1, \mathbf{g}_2 \rangle = (2.0 \times -1.0) + (1.0 \times 3.0) = -2.0 + 3.0 = +1.0$$

**Step 2: Compute difference vector $(\mathbf{g}_2 - \mathbf{g}_1)$:**
$$\mathbf{g}_2 - \mathbf{g}_1 = \begin{bmatrix} -1.0 - 2.0 \\ 3.0 - 1.0 \end{bmatrix} = \begin{bmatrix} -3.0 \\ 2.0 \end{bmatrix}$$

**Step 3: Compute squared Euclidean norm $\|\mathbf{g}_1 - \mathbf{g}_2\|^2$:**
$$\|\mathbf{g}_1 - \mathbf{g}_2\|^2 = (-3.0)^2 + (2.0)^2 = 9.0 + 4.0 = 13.0$$

**Step 4: Compute numerator $(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2$:**
$$(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2 = (-3.0 \times -1.0) + (2.0 \times 3.0) = 3.0 + 6.0 = 9.0$$

**Step 5: Solve for optimal weight $\alpha^*$:**
$$\alpha^* = \text{clip}\left( \frac{9.0}{13.0}, 0, 1 \right) = \text{clip}(0.6923, 0, 1) = 0.6923$$
$$\beta^* = 1 - \alpha^* = 1 - 0.6923 = 0.3077$$
**Conclusion:** The optimal common descent gradient is:
$$\mathbf{g}_{\text{MGDA}} = 0.6923 \mathbf{g}_1 + 0.3077 \mathbf{g}_2 = \begin{bmatrix} 0.6923(2) + 0.3077(-1) \\ 0.6923(1) + 0.3077(3) \end{bmatrix} = \begin{bmatrix} 1.3846 - 0.3077 \\ 0.6923 + 0.9231 \end{bmatrix} = \begin{bmatrix} 1.0769 \\ 1.6154 \end{bmatrix}$$

---

# SECTION 3: PRESENTATION SCRIPTS & HANDOVERS

## 3.1 Five-Minute Presentation Script (Member 2)

> "Thank you, Member 1. Respected examiners, I am Member 2, leading the Deep Learning, Multi-Objective Optimization, and Region-Adaptive Compression development.
> 
> As Member 1 demonstrated, standard uniform JPEG compression faces an aggressive trade-off: compressing an entire ultrasound image degrades delicate acoustic margins, directly impairing computerized segmentation.
> 
> To perform clinical lesion delineation on the receiving hospital workstation, I implemented a 4-stage biomedical U-Net in `models/unet.py` containing 7.76 million parameters. Skip connections bridge high-resolution encoder features directly across to the decoder, ensuring sub-pixel boundary localization.
> 
> To train this network, combining Binary Cross-Entropy and Dice loss with arbitrary fixed weights leads to destructive gradient interference. To solve this, I integrated the Multi-Gradient Descent Algorithm (MGDA) in `training/mgda_solver.py`. By finding the Frank-Wolfe minimum norm on the Pareto gradient frontier, MGDA dynamically calculates the optimal loss weighting $\alpha^*$ at every step, boosting our validation Dice score from $0.6056$ to $0.6581$, a direct $+5.25\%$ clinical gain.
> 
> To enable region-adaptive compression at the edge sender without a dedicated GPU, I engineered a compact ROI Detector in `models/roi_detector.py`. With only 482,449 parameters—over 16 times smaller than our clinical U-Net—it executes on a standard laptop CPU in just 3.8 milliseconds.
> 
> In `sender/adaptive_compressor.py`, I combined this detector with morphological dilation using an elliptical structuring element and Gaussian feathering. This preserves the lesion and a 15-pixel margin at Q90 while compressing non-diagnostic background at Q20. In physical testing on case `benign (104)`, this achieved a 37.9% byte reduction over uniform Q90 while retaining an identical Dice score of $0.8664$.
> 
> I now hand over to Member 3, who will explain our TCP network transmission architecture, Linux netem channel emulation, and physical two-laptop telemetry."

---

## 3.2 Ten-Minute Extended Presentation Script (Member 2)

> *(Use for in-depth technical presentations)*
> 
> "Respected evaluators, good morning. My name is Member 2, and I am responsible for Deep Learning Architectures, Multi-Objective Loss Optimization, and Region-Adaptive Compression Engineering.
> 
> **1. Clinical Segmentation Engine (U-Net):**
> On the receiver hospital workstation, our core diagnostic task is automated semantic lesion segmentation. In `models/unet.py`, I built a PyTorch implementation of the classic biomedical U-Net. The network takes a normalized $256 \times 256 \times 1$ tensor and processes it through a 4-stage encoder-decoder pipeline with channel depths scaling from 64 to 512, reaching a 1024-channel bottleneck. The complete model contains $7,762,465$ parameters ($31.2\text{ MB}$).
> 
> Lateral skip connections are critical here: they concatenate high-resolution spatial feature maps from the contracting path into the expanding path. This counters the spatial information loss caused by max-pooling downsampling, allowing the $1 \times 1$ convolutional output layer with sigmoid activation to delineate intricate, irregular tumor contours.
> 
> **2. Multi-Objective Pareto Optimization via MGDA:**
> A major research contribution of our project is solving the multi-objective loss conflict during U-Net training. In medical imaging, Binary Cross-Entropy loss evaluates global pixel accuracy, but it suffers from severe class imbalance because lesions typically occupy less than 5% of the image area. Dice loss focuses directly on foreground overlap, but its non-convex optimization surface causes unstable gradients.
> 
> Standard approaches sum them with arbitrary scalar weights like $0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$. However, when the gradients $\mathbf{g}_1$ and $\mathbf{g}_2$ point in opposite directions, scalar summation causes destructive gradient cancellation.
> 
> In `training/mgda_solver.py`, I formulated this as a multi-objective optimization problem solved via the Multi-Gradient Descent Algorithm (MGDA). Using Karush-Kuhn-Tucker conditions and the Frank-Wolfe algorithm, my solver computes the exact analytical minimum norm projection:
> $$\alpha^* = \text{clip}\left(\frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1\right)$$
> Applying this common descent vector improved our best validation Dice score from $0.6056$ (achieved at Epoch 21 under baseline training) to $0.6581$ (achieved at Epoch 19 under MGDA), an absolute clinical gain of $+5.25\%$.
> 
> **3. Edge ROI Detector Architecture:**
> While our 7.76M parameter U-Net is ideal for high-powered server inference, it is too heavy for edge transmission nodes. To solve this, I designed a specialized, lightweight ROI Detector in `models/roi_detector.py`.
> 
> By scaling encoder channels to $[16, 32, 64, 128]$ and bottleneck channels to 256, I reduced the parameter count to 482,449—making it $16.1\times$ smaller than the clinical U-Net, occupying just 1.96 MB. On an Intel CPU, it achieves an inference latency of only 3.82 milliseconds. Evaluated on the 117 quarantined held-out test scans, it achieves a 0.6087 Dice score and 0.5126 IoU, providing fast, dependable edge localization.
> 
> **4. Region-Adaptive Compression Engine:**
> In `sender/adaptive_compressor.py`, I integrated the ROI detector into our region-adaptive compression engine. The challenge with region-based compression is boundary edge artifacts: if you cut and paste sharp rectangular blocks, artificial $90^\circ$ boundaries trigger DCT ringing and false convolutional activations.
> 
> My pipeline executes four mathematical stages:
> First, it expands the predicted lesion mask using morphological dilation with an elliptical structuring element (`cv2.MORPH_ELLIPSE`) by a configurable margin $m$ (default 15 pixels).
> Second, it applies Gaussian feathering with $\sigma = m/3 = 5.0$ to produce a continuously graded alpha mask $A(x, y) \in [0.0, 1.0]$.
> Third, it compresses dual full-frame representations at Q90 ($I_{\text{high}}$) and Q20 ($I_{\text{low}}$).
> Fourth, it blends them via spatial alpha compositing: $I_{\text{adaptive}} = A \cdot I_{\text{high}} + (1 - A) \cdot I_{\text{low}}$, and packages the output into a compliant JPEG stream.
> 
> In physical trials recorded in `experiments/experiments.sqlite`, case `benign (104)` dropped in size from $78,050\text{ bytes}$ at uniform Q90 down to $48,481\text{ bytes}$ under Adaptive compression—a $37.9\%$ byte reduction—while maintaining the exact same clinical segmentation Dice score of $0.8664$.
> 
> I now hand over to Member 3 to explain our network transport protocol, wire clocking telemetry, and physical two-laptop deployment."

---

# SECTION 4: 50 DEDICATED VIVA QUESTIONS & DEFENSE (MEMBER 2)

### Q1: What is the exact parameter count and disk size of your clinical U-Net model?
- **Direct Answer:** Our clinical U-Net in `models/unet.py` has exactly **7,762,465** trainable parameters, occupying **31.2 MB** on disk (`models/unet_busi.pth`).

### Q2: What is the exact parameter count and disk size of your edge ROI Detector?
- **Direct Answer:** Our edge ROI Detector in `models/roi_detector.py` has exactly **482,449** trainable parameters, occupying **1.96 MB** on disk (`models/roi_detector_v1.pth`), which is **16.1 times smaller** than the clinical U-Net.

### Q3: Why did you build two separate models instead of using U-Net on both sender and receiver?
- **Direct Answer:** Asymmetry in edge vs server compute constraints. The sender runs on low-power edge hardware (laptop CPU) where a 7.76M parameter model takes ~80 ms and risks frame stuttering. The ROI detector executes in 3.8 ms (20× faster) with minimal RAM overhead, leaving high-precision clinical segmentation to the receiver workstation.

### Q4: Explain the role of skip connections in U-Net.
- **Direct Answer:** Skip connections copy high-resolution spatial feature maps from the encoder stages and concatenate them directly with the corresponding decoder feature maps.
- **Technical Detail:** Downsampling via max-pooling destroys spatial coordinate precision. Skip connections bypass the bottleneck, directly propagating fine-grained spatial gradients and localization cues to the final segmentation layers.

### Q5: What is the mathematical definition of Binary Cross-Entropy (BCE) loss?
- **Direct Answer:**
  $$\mathcal{L}_{\text{BCE}}(y, \hat{p}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
  where $y_i \in \{0, 1\}$ is the ground truth and $\hat{p}_i = \sigma(z_i) \in [0, 1]$ is the predicted probability.

### Q6: What is the mathematical definition of Dice Loss?
- **Direct Answer:**
  $$\mathcal{L}_{\text{Dice}}(Y, \hat{P}) = 1 - \frac{2 \sum_{i=1}^{N} y_i \hat{p}_i + \epsilon}{\sum_{i=1}^{N} y_i + \sum_{i=1}^{N} \hat{p}_i + \epsilon}$$
  where $\epsilon = 10^{-6}$ is a smoothing factor to prevent division by zero.

### Q7: Why does BCE loss perform poorly on its own in breast ultrasound segmentation?
- **Direct Answer:** Extreme class imbalance. In a typical $256 \times 256$ ultrasound scan, the lesion occupies only 2% to 5% of the total pixel area. A model that predicts all background zeros achieves 95% to 98% BCE accuracy while having 0% clinical utility.

### Q8: Why does Dice loss perform poorly on its own?
- **Direct Answer:** Dice loss has a highly non-convex loss surface with vanishing gradients in early training iterations when predicted probabilities $\hat{p}_i$ are near zero, often causing optimization to become trapped in bad local minima.

### Q9: What is the problem with using a simple fixed-weight combined loss like $0.5\text{BCE} + 0.5\text{Dice}$?
- **Direct Answer:** Gradient conflict. The gradient vectors $\nabla \mathcal{L}_{\text{BCE}}$ and $\nabla \mathcal{L}_{\text{Dice}}$ frequently point in opposing directions ($\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0$). Fixed scalar addition forces one loss to undo the progress of the other.

### Q10: What is Pareto stationarity in multi-objective optimization?
- **Direct Answer:** A point $\theta^*$ is Pareto-stationary if no common descent direction exists that can decrease one objective without increasing at least one other objective. Mathematically, $\mathbf{0} \in \text{conv}(\{\nabla \mathcal{L}_1(\theta^*), \dots, \nabla \mathcal{L}_m(\theta^*)\})$.

### Q11: State the closed-form formula for the optimal weight $\alpha^*$ in 2-objective MGDA.
- **Direct Answer:**
  $$\alpha^* = \text{clip}\left( \frac{(\mathbf{g}_2 - \mathbf{g}_1)^T \mathbf{g}_2}{\|\mathbf{g}_1 - \mathbf{g}_2\|^2}, 0, 1 \right)$$
- **Code Reference:** `training/mgda_solver.py:MGDASolver.find_min_norm_element`

### Q12: What empirical gain did MGDA optimization achieve over baseline training?
- **Direct Answer:** MGDA boosted the best validation Dice score from **0.6056** (Epoch 21) to **0.6581** (Epoch 19), representing a **+5.25% absolute clinical improvement**.

### Q13: What was the held-out test performance of the lightweight ROI Detector?
- **Direct Answer:** Evaluated strictly on the 117 quarantined held-out test scans from `data/BUSI/splits.json`, it achieved a Test Dice of **0.6087** and Test IoU of **0.5126**.

### Q14: What is the CPU inference latency of the ROI Detector versus the clinical U-Net?
- **Direct Answer:** The ROI Detector runs in **3.82 ms** on a standard CPU, whereas the clinical U-Net requires **79.39 ms** (a $20.8\times$ speedup).

### Q15: Explain the 4 steps of Region-Adaptive Compression in `sender/adaptive_compressor.py`.
- **Direct Answer:**
  1. Detect lesion ROI using `ROIDetector` or mask input.
  2. Dilate mask with elliptical kernel (`cv2.MORPH_ELLIPSE`) by margin $m$ (15 px).
  3. Apply Gaussian blur ($\sigma = m/3$) to create continuous alpha blend mask $A(x, y) \in [0, 1]$.
  4. Spatially blend Q90 image and Q20 image: $I_{\text{adaptive}} = A \cdot I_{\text{Q90}} + (1 - A) \cdot I_{\text{Q20}}$, then encode as standard JPEG.

### Q16: Why is Gaussian feathering ($\sigma = m/3$) necessary?
- **Direct Answer:** Abrupt binary cutoffs create sharp rectangular edges that generate high-frequency Fourier components, causing severe Gibbs ringing artifacts during JPEG DCT quantization and triggering false edge detections in convolutional neural networks.

### Q17: What was the byte reduction achieved by Region-Adaptive compression on `benign (104)`?
- **Direct Answer:** File size dropped from **78,050 bytes** (uniform Q90) to **48,481 bytes** (Adaptive Q90/Q20), achieving a **37.9% byte reduction**.

### Q18: Did Region-Adaptive compression degrade the segmentation Dice score on `benign (104)`?
- **Direct Answer:** No. Both uniform Q90 and Region-Adaptive compression produced an identical Dice score of **0.8664** and IoU of **0.7643** (0.0000 degradation).

### Q19: What is the mathematical relationship between Dice and IoU?
- **Direct Answer:**
  $$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

### Q20: Which metric is always lower: Dice or IoU?
- **Direct Answer:** IoU is always less than or equal to Dice ($\text{IoU} \le \text{Dice}$). They are equal only when both are 0.0 or 1.0.

### Q21: What is the Sigmoid activation formula and what threshold $\tau$ was used?
- **Direct Answer:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  A standard clinical threshold of $\tau = 0.5$ is applied: $\hat{y} = 1$ if $\sigma(z) \ge 0.5$, else $0$.

### Q22: Why use Batch Normalization in U-Net?
- **Direct Answer:** Batch Normalization standardizes intermediate activations ($\mu=0, \sigma^2=1$), reducing internal covariate shift, smoothing the optimization landscape, and accelerating convergence.

### Q23: What optimizer, learning rate, and batch size were used during training?
- **Direct Answer:** Adam optimizer with initial learning rate $\eta = 10^{-4}$, weight decay $10^{-5}$, and batch size 16.

### Q24: What is the purpose of early stopping and what was the patience?
- **Direct Answer:** Early stopping halts training when validation loss stops improving to prevent overfitting. We used a patience of 10 epochs monitoring validation Dice.

### Q25: How does your receiver display segmentation overlays without visual distortion?
- **Direct Answer:** In `ui/receiver/segmentation_view.py`, the predicted $256 \times 256$ mask is resized back to the original image's uncropped dimensions ($777 \times 578$) using `cv2.INTER_NEAREST` before generating the green semi-transparent contour overlay, maintaining exact 1:1 aspect ratio with the received scan.

### Q26: What is the difference between semantic segmentation and object detection?
- **Direct Answer:** Object detection outputs coarse bounding boxes $(x, y, w, h)$ identifying object locations. Semantic segmentation classifies every individual pixel into a semantic category, outlining exact morphological boundaries.

### Q27: What is the difference between Transposed Convolution and Bilinear Upsampling?
- **Direct Answer:** Transposed convolution has learnable kernel parameters that adapt during backpropagation, while bilinear upsampling uses fixed, deterministic mathematical interpolation with zero learnable parameters.

### Q28: How do you handle an empty ground-truth mask in `compute_dice_score`?
- **Direct Answer:** If both ground truth and prediction have zero foreground pixels, we return 1.0 (perfect true negative). If ground truth is zero but prediction has foreground pixels, we return 0.0 (false positive hallucination).

### Q29: What is the purpose of an elliptical structuring element over a rectangular one?
- **Direct Answer:** Biological tumors expand radially, forming curved organic shapes. Elliptical dilation maintains smooth curved contours, avoiding artificial $90^\circ$ vertices that disrupt DCT frequency spectra.

### Q30: What happens if the ROI Detector fails and predicts an empty mask for a lesion?
- **Direct Answer:** In `sender/adaptive_compressor.py`, if the predicted mask is completely empty ($|\hat{M}| = 0$), the compressor triggers an automatic safety fallback to uniform Q70 compression to prevent lesion loss.

### Q31: How many channels does the input tensor have in `UNet`?
- **Direct Answer:** Exactly 1 channel (`in_channels=1`), corresponding to single-channel 8-bit normalized grayscale ultrasound images.

### Q32: What is the purpose of the bottleneck layer in U-Net?
- **Direct Answer:** The bottleneck layer (1024 channels at $16 \times 16$ spatial resolution) compresses the receptive field to capture global semantic context—determining the overall acoustic texture and deep feature relationships.

### Q33: How does the number of parameters in a convolutional layer scale?
- **Direct Answer:** $\text{Params} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$, where $K_h, K_w$ are kernel dimensions, $C_{\text{in}}$ is input channels, $+1$ is bias, and $C_{\text{out}}$ is output channels.

### Q34: What is the receptive field of two stacked $3 \times 3$ convolutions?
- **Direct Answer:** Two stacked $3 \times 3$ convolutions have an effective receptive field of $5 \times 5$, but with fewer parameters ($2 \times 3^2 = 18$ weights vs $5^2 = 25$ weights) and an extra non-linear activation.

### Q35: What is Frank-Wolfe algorithm in the context of MGDA?
- **Direct Answer:** Frank-Wolfe is an iterative projection-free convex optimization algorithm. In MGDA, it solves the quadratic subproblem of finding the minimum-norm vector in the convex hull of task gradients.

### Q36: What is a Pareto front?
- **Direct Answer:** The set of all Pareto-optimal solutions where no single objective can be improved without degrading at least one other objective.

### Q37: What is the default margin parameter $m$ in your system?
- **Direct Answer:** The default margin is **15 pixels** (`margin_px=15`), providing a robust spatial buffer around detected tumor margins.

### Q38: What are the default ROI and background quality factors in your system?
- **Direct Answer:** $Q_{\text{ROI}} = 90$ and $Q_{\text{BG}} = 20$.

### Q39: Why not set background quality to Q10 or Q5?
- **Direct Answer:** Below Q20, severe blocking artifacts bleed across the Gaussian blend zone, and secondary anatomical reference landmarks (like Cooper's ligaments and pectoral muscle) become completely obliterated, which disorients radiologists.

### Q40: In `models/unet.py`, how are tensors concatenated along the skip connection?
- **Direct Answer:** `torch.cat([skip_connection, upsampled_tensor], dim=1)` concatenates feature maps along the channel dimension.

### Q41: Why did you freeze the U-Net weights on the receiver node during experiments?
- **Direct Answer:** Freezing weights ensures scientific rigor and reproducibility. The model acts as an invariant, objective clinical benchmark to measure purely the distortion introduced by different compression strategies.

### Q42: What is the role of `torch.no_grad()` in `receiver/segmentation_engine.py`?
- **Direct Answer:** `torch.no_grad()` disables PyTorch's autograd tracking engine during inference, saving memory allocation and accelerating CPU/GPU execution speed.

### Q43: What is the difference between soft Dice loss and hard Dice score?
- **Direct Answer:** Soft Dice uses continuous predicted probabilities $\hat{p} \in [0, 1]$ directly in backpropagation so derivatives can flow. Hard Dice applies a binary threshold $\tau = 0.5$ yielding discrete binary sets $\{0, 1\}$ for evaluation.

### Q44: Can the U-Net handle images with arbitrary aspect ratios?
- **Direct Answer:** While U-Net is fully convolutional and can theoretically process arbitrary dimensions, our standardized pipeline resizes inputs to $256 \times 256$ for batch processing consistency, then upsamples the predicted mask back to the native aspect ratio for display.

### Q45: What is the loss formula used in `training/train_mgda.py`?
- **Direct Answer:** $\mathcal{L}_{\text{total}} = \alpha^* \mathcal{L}_{\text{BCE}} + (1 - \alpha^*) \mathcal{L}_{\text{Dice}}$, where $\alpha^*$ is dynamically computed by `MGDASolver`.

### Q46: How does the system ensure reproducibility across training runs?
- **Direct Answer:** We set seeds for Python `random.seed(42)`, NumPy `np.random.seed(42)`, and PyTorch `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)`.

### Q47: What does an IoU score of 0.7643 indicate clinically?
- **Direct Answer:** In clinical computer-aided diagnosis literature, an IoU $>0.70$ (corresponding to a Dice $>0.82$) is classified as high clinical concordance, providing boundary accuracy sufficient for biopsy localization and volume tracking.

### Q48: How is the alpha mask clamped in `sender/adaptive_compressor.py`?
- **Direct Answer:** Using `np.clip(alpha, 0.0, 1.0).astype(np.float32)`.

### Q49: Does the sender transmit the alpha mask to the receiver?
- **Direct Answer:** No. The sender transmits only the final blended JPEG image. The receiver does not need the alpha mask, keeping the network protocol lightweight and 100% compliant with standard JPEG decoders.

### Q50: How do you know the U-Net did not overfit?
- **Direct Answer:** We tracked validation Dice curves, applied early stopping with patience 10, utilized dropout and weight decay ($10^{-5}$), and verified generalization on our 117 quarantined held-out test scans.

---

# SECTION 5: SUMMARY CHECKLIST FOR MEMBER 2

Before stepping into the evaluation room, ensure you can:
- [ ] Draw the U-Net architecture on a whiteboard, labeling channel counts at every stage ($64 \to 128 \to 256 \to 512 \to 1024$).
- [ ] Write down the formulas for BCE loss, Dice loss, and the MGDA $\alpha^*$ closed-form solution.
- [ ] State the parameter counts from memory: U-Net ($7,762,465$), ROI Detector ($482,449$).
- [ ] State the exact validation Dice gain achieved by MGDA: from $0.6056$ to $0.6581$ ($+5.25\%$).
- [ ] State the physical adaptive compression result for `benign (104)`: $37.9\%$ byte reduction, exact Dice $0.8664$.
- [ ] Perform manual 4-cell Dice and IoU calculations on paper.
- [ ] Explain why Gaussian feathering eliminates artificial edge artifacts.
- [ ] Deliver your 5-minute presentation script fluently without notes.
- [ ] Seamlessly hand over to Member 3.
