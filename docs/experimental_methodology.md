# Academic Research Methodology & Experimental Protocol

**Project Title:** Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation  
**Context:** Multi-Node Cross-Platform Medical Tele-ultrasound Framework

---

## 1. Experimental Objectives

This research systematically examines the clinical trade-offs between network bandwidth conservation, transmission latency, image quality degradation, and deep learning lesion segmentation accuracy in tele-ultrasound systems:

1. **Rate-Distortion Characterization**: Quantify mathematical degradation across JPEG compression qualities ($Q \in \{30, 50, 70, 90\}$).
2. **Channel Emulation**: Measure socket transit latency under hardware-enforced bandwidth, delay, and packet loss constraints.
3. **Segmentation Robustness**: Measure Dice and IoU metric degradation on frozen U-Net predictions across compression ratios.
4. **Adaptive Rate Allocation**: Validate a dynamic rate controller capable of choosing compression levels based on estimated channel conditions.

---

## 2. Quantitative Metrics Formulation

### 2.1 Peak Signal-to-Noise Ratio (PSNR)
PSNR assesses the pixel-level fidelity of the compressed image $I_c$ relative to the original image $I_o$:

$$\text{MSE} = \frac{1}{M \cdot N} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I_o(i, j) - I_c(i, j)]^2$$

$$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$

where $\text{MAX}_I = 255$ for 8-bit grayscale ultrasound images. If $I_o = I_c$, $\text{PSNR} = \infty$.

### 2.2 Structural Similarity Index (SSIM)
SSIM evaluates perceptual structural degradation by comparing luminance, contrast, and structure:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Constants $C_1 = (0.01 \cdot 255)^2$ and $C_2 = (0.03 \cdot 255)^2$ prevent numerical instability.

### 2.3 Dice Similarity Coefficient (DSC)
Quantifies spatial overlap between the binary predicted lesion mask $P$ and ground-truth mask $G$:

$$\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

### 2.4 Intersection over Union (IoU / Jaccard Index)
$$\text{IoU}(P, G) = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{\text{Dice}}{2 - \text{Dice}}$$

---

## 3. Network Emulation Profiles

Linux `tc` (Traffic Control) and `netem` (Network Emulator) shape egress queues directly on the bridged Ethernet interface:

| Profile | Bandwidth | Added Delay (Round-Trip) | Packet Loss | Target Clinical Tele-Medicine Scenario |
| :--- | :--- | :--- | :--- | :--- |
| **High Speed** | 10 Mbps | 5 ms | 0.0% | High-speed dedicated hospital local Ethernet |
| **Medium Speed** | 2 Mbps | 20 ms | 0.5% | Suburban broadband / rural regional health clinic |
| **Constrained** | 512 Kbps | 50 ms | 1.0% | 3G / Congested cellular medical outpost |
| **Severely Constrained** | 128 Kbps | 100 ms | 2.0% | Emergency satellite uplink / disaster mobile unit |

### Linux Commands Executed:
```bash
# Egress profile application
sudo tc qdisc add dev <iface> root netem rate <bw> delay <delay>ms loss <loss>%

# Verification
tc qdisc show dev <iface>

# Reset to unshaped
sudo tc qdisc del dev <iface> root
```

---

## 4. Adaptive Compression Controller Logic

The adaptive controller dynamically allocates JPEG quality according to available channel bandwidth:

$$\text{Quality}(B) = \begin{cases} 
90 & \text{if } B \ge 10\text{ Mbps (Tier 1: High Quality)} \\ 
70 & \text{if } 2\text{ Mbps} \le B < 10\text{ Mbps (Tier 2: Balanced)} \\ 
50 & \text{if } 512\text{ Kbps} \le B < 2\text{ Mbps (Tier 3: Constrained)} \\ 
30 & \text{if } B < 512\text{ Kbps (Tier 4: Severely Constrained)} 
\end{cases}$$

### Dynamic Bandwidth Tracking (EWMA):
When configured in measured mode, available bandwidth is updated dynamically from genuine socket throughput measurements:

$$\hat{B}_t = \alpha \cdot M_t + (1 - \alpha) \cdot \hat{B}_{t-1}$$

where $\alpha = 0.30$ (configurable in the GUI), ensuring stability against transient socket burst jitter.

---

## 5. Frozen U-Net Deep Learning Pipeline

To preserve experimental reproducibility:
1. Model weights are mounted in strictly **frozen evaluation mode** (`model.eval()`, `requires_grad=False`).
2. Input images are decoded from transmitted JPEG bytes, converted to 1-channel grayscale, resized to $256 \times 256$ via bilinear interpolation, and normalized to float32 $[0.0, 1.0]$.
3. Sigmoidal activation is binarized at $\tau = 0.50$ to produce the binary predicted lesion mask.
4. Comparative overlays highlight classification regions:
   - **Yellow**: True Positives ($P \cap G$)
   - **Red**: False Positives ($P \setminus G$)
   - **Green**: False Negatives ($G \setminus P$)
