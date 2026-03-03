# 📘 Theory: Concepts in the Change Detection Project

---

# 📑 Contents

1. Notation  
2. Problem Statement  
3. Data Pairing, Alignment & Preprocessing  
4. Input Strategies  
5. Model Architecture: Siamese U-Net  
6. Loss Functions (Mathematical Formulation)  
7. Metrics & Evaluation Protocol  
8. Training Best Practices  
9. Post-processing & Visualization  
10. Advanced Extensions  
11. Practical Diagnostics  
12. References  

---

# 1️⃣ Notation

Let:

- $A, B \in \mathbb{R}^{C \times H \times W}$  
  Satellite images at times $t_1$ and $t_2$  
  (typically $C = 3$ for RGB)

- Ground truth change mask:
  $$
  Y \in \{0,1\}^{H \times W}
  $$

- Model logits:
  $$
  L(A,B) \in \mathbb{R}^{1 \times H \times W}
  $$

- Probability map:
  $$
  P = \sigma(L) \in [0,1]^{H \times W}
  $$

- Sigmoid function:
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$

- Binary prediction at threshold $\tau$:
  $$
  P_\tau = \mathbf{1}_{P > \tau}
  $$

- Pixel index set:
  $$
  \Omega = \{1,...,H\} \times \{1,...,W\}
  $$

---

# 2️⃣ Problem Statement

Given a **co-registered image pair** $(A, B)$:

Estimate a binary mask $Y$ such that:

$$
Y_i =
\begin{cases}
1 & \text{if pixel } i \text{ changed between } t_1 \text{ and } t_2 \\
0 & \text{otherwise}
\end{cases}
$$

This is a **per-pixel binary segmentation problem**.

---

# 3️⃣ Data Pairing, Alignment & Preprocessing

### 🔹 Pairing
Ensure:
- $A[i]$, $B[i]$, and $Y[i]$ are spatially aligned  
- Same geographic viewport  
- Same spatial resolution  

### 🔹 Registration
If misalignment exists:
- Apply geometric registration  
  - Rigid  
  - Affine  
  - Non-rigid  

### 🔹 Normalization

Per-channel standardization:

$$
x'_{c,i} = \frac{x_{c,i} - \mu_c}{\sigma_c}
$$

Where:
- $\mu_c$ and $\sigma_c$ estimated from training data

### 🔹 Mask Binarization

$$
Y' = \mathbf{1}_{Y > 0.5}
$$

### 🔹 Optional Remote-Sensing Preprocessing

- Histogram matching  
- Radiometric correction  
- Cloud masking  

---

# 4️⃣ Input Strategies

## 🔹 1. Concatenation

$$
X = \text{concat}(A,B) \in \mathbb{R}^{2C \times H \times W}
$$

Directly fed to a segmentation network.

---

## 🔹 2. Siamese Strategy (Used in This Project)

Shared encoder $E$:

$$
F_A = E(A), \quad F_B = E(B)
$$

Feature combination:

- Absolute difference:
  $$
  D = |F_A - F_B|
  $$

- Or signed subtraction:
  $$
  S = F_B - F_A
  $$

Decoder maps $D \rightarrow$ predicted mask.

### 🎯 Rationale

Feature differencing:
- Suppresses invariant structures  
- Highlights change-specific signals  

---

# 5️⃣ Model Architecture: Siamese U-Net

## 🔹 Encoder

Repeated blocks:

Conv → BatchNorm → ReLU → Pool

At stage $s$:

$$
f_s \in \mathbb{R}^{C_s \times H_s \times W_s}
$$

For each image:

$$
f_s^A, \quad f_s^B
$$

Feature difference:

$$
d_s = | f_s^A - f_s^B |
$$

---

## 🔹 Decoder

- Upsampling (Transposed Conv or Bilinear + Conv)
- Skip connections
- Feature refinement

Final output:

$$
L = \text{Conv}_{1\times1}(\text{decoder output})
$$

Where:

$$
L \in \mathbb{R}^{1 \times H \times W}
$$

---

# 6️⃣ Loss Functions

## 🔹 Sigmoid

$$
p_i = \sigma(L_i)
$$

---

## 🔹 Binary Cross-Entropy (BCE)

$$
\text{BCE} =
- \frac{1}{|\Omega|}
\sum_{i \in \Omega}
\left[
y_i \log p_i + (1-y_i)\log(1-p_i)
\right]
$$

---

## 🔹 Dice Coefficient

$$
\text{Dice} =
\frac{2 \sum_i p_i y_i}
{\sum_i p_i + \sum_i y_i + \epsilon}
$$

---

## 🔹 Dice Loss

$$
L_{\text{Dice}} = 1 - \text{Dice}
$$

---

## 🔹 Combined Loss (Used in Project)

$$
L = \alpha \cdot \text{BCE}
+
\beta \cdot L_{\text{Dice}}
$$

Typically:

$$
\alpha = \beta = 0.5
$$

---

# 7️⃣ Metrics & Evaluation

Define:

- True Positives:
  $$
  TP = \sum_i P_{\tau,i} \cdot y_i
  $$

- False Positives:
  $$
  FP = \sum_i P_{\tau,i}(1-y_i)
  $$

- False Negatives:
  $$
  FN = \sum_i (1-P_{\tau,i})y_i
  $$

---

## 🔹 Intersection over Union (IoU)

$$
\text{IoU} =
\frac{TP}{TP + FP + FN + \epsilon}
$$

---

## 🔹 F1 Score

$$
F1 =
\frac{2TP}{2TP + FP + FN + \epsilon}
$$

---

## 🔹 Evaluation Protocol

- Use `@torch.no_grad()` during validation/test
- Apply sigmoid → threshold
- Compute per-image IoU and F1
- Report mean over dataset

---

# 8️⃣ Training Best Practices

- Use GPU if available (`device="cuda"`)
- Choose largest feasible batch size
- Optimizer: AdamW or SGD + weight decay
- Use learning rate scheduling
- Apply data augmentation:
  - Flips
  - Rotations
  - Crops
  - Color jitter
- Monitor:
  - Training loss
  - Validation IoU
  - Validation F1
- Use early stopping or checkpointing

---

# 9️⃣ Post-processing & Visualization

- Threshold tuning
- Precision-Recall analysis
- Morphological opening/closing
- Remove small connected components
- Contour overlays for qualitative inspection
- Save probability maps for calibration

---

# 🔟 Advanced Extensions

- Multi-spectral & SAR adaptation
- Multi-temporal sequence modeling
- Attention mechanisms
- Uncertainty estimation (MC Dropout, Ensembles)
- Domain adaptation
- Focal / Tversky loss for imbalance

---

# 1️⃣1️⃣ Practical Diagnostics

- Visualize intermediate feature differences $d_s$
- Inspect false positives and negatives
- Check alignment issues
- Reduce LR if training diverges
- Fix random seeds for reproducibility

---

# 📚 References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)  
- Siamese network literature  
- Remote sensing change detection research papers  

---

# ✅ Summary

This project implements a complete **supervised deep learning framework** for satellite image change detection:

- Mathematically grounded  
- Architecturally structured  
- Evaluation-driven  
- Extendable to research-level applications  