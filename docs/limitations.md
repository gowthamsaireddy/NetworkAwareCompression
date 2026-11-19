# Research Limitations & Clinical Disclaimer

**Project**: Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation  
**Document**: `docs/limitations.md`  
**Compliance**: Clinical Research Ethics & Engineering Boundary Analysis

---

## 1. Regulatory & Clinical Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL RESEARCH PROTOTYPE ONLY**  
> This software system is an academic research prototype engineered to evaluate the interplay between network bandwidth constraints, lossy compression algorithms, and deep neural network segmentation inference. It has **NOT** been cleared, certified, or approved by the U.S. FDA, European CE Mark, or any medical regulatory agency. It is **NOT** intended for clinical diagnosis, patient triage, treatment planning, or commercial telemedicine deployment without formal clinical trial validation and certified software lifecycle compliance (IEC 62304 / ISO 13485).

---

## 2. Dataset Limitations (BUSI)

1. **Sample Size & Geographic Diversity**:
   - The Breast Ultrasound Images (BUSI) dataset consists of 780 ultrasound images acquired from 600 female patients at Baheya Hospital, Cairo, Egypt (LOGIQ E9 and LOGIQ 7 ultrasound systems).
   - While representative of this cohort, the dataset lacks geographic and ethnic demographic diversity.
2. **Resolution Inhomogeneity**:
   - Original ultrasound captures range from $340 \times 450$ to $700 \times 800$ pixels.
   - Rescaling to target $256 \times 256$ input tensors for U-Net processing introduces geometric resampling interpolation, slightly smoothing micro-calcifications and subtle acoustic boundaries.
3. **Class Imbalance**:
   - BUSI contains 437 benign cases, 210 malignant cases, and 133 normal cases.
   - Benign cases outnumber malignant cases by more than $2:1$, requiring balanced metrics reporting (Dice, IoU, Specificity, and Sensitivity).

---

## 3. Compression Artifacts & Lesion Morphology

1. **High-Frequency Attenuation at Low Bitrates**:
   - Standard discrete cosine transform (DCT) based JPEG compression partitions images into $8 \times 8$ pixel blocks.
   - At aggressive compression rates (e.g. $Q \le 30$), coarse quantization truncates high-frequency DCT coefficients, yielding block-boundary discontinuities ("blocking artifacts") and blur across specular lesion boundaries.
2. **Acoustic Shadowing Distortion**:
   - Malignant ultrasound lesions typically exhibit posterior acoustic shadowing (attenuation of ultrasound beam beneath dense fibrotic lesions).
   - Compression algorithms can misquantize low-luminance acoustic shadow gradients as flat background, altering posterior boundary definition.
3. **Region-Adaptive Feathering Trade-Off**:
   - While our Gaussian alpha-feathered margin ($\sigma = m / 3.0$) eliminates step-edge artifacts, selecting an excessively narrow margin ($m < 8\text{ px}$) risks clipping infiltrative margins in irregularly shaped, non-circumscribed malignant masses.

---

## 4. Network Emulation vs. Physical Cellular/Satellite Reality

1. **Synthetic Netem Constraints**:
   - Emulation via Linux `tc/netem` provides controlled, reproducible testing (static rate, constant delay, uniform packet drop probability).
   - Real-world wireless telemedicine environments (4G/5G mobile health vans, rural satellite clinics) exhibit non-stationary multi-path fading, burst packet drops, Doppler shift, and dynamic TCP congestion collapse (bufferbloat).
2. **Socket Buffer Dynamics**:
   - Measurements of pure wire transfer time capture kernel socket queue push rates. In high-latency satellite links ($>600\text{ ms}$ RTT), TCP sliding window scaling dominates overall application goodput.

---

## 5. Hardware Dependencies & Timing Precision

1. **Clock Monotonicity & OS Scheduling**:
   - High-resolution timing utilizes Python `time.perf_counter()`, which is monotonic. However, non-real-time operating systems (Windows and standard Linux kernels) are subject to thread context switches, OS interrupts, and GPU driver kernel launch latencies.
2. **GPU Cold-Start Overhead**:
   - First-pass inference on CUDA incurs CUDA runtime initialization and cuDNN kernel selection overhead (~1.5s cold start). Subsequent passes execute in steady-state ($< 25\text{ ms}$). Steady-state measurements are used for reporting.
