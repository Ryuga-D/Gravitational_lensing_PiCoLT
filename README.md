# Physics-Informed Continuous Lens Transformer (PICoLT)

> A deep learning framework for **simultaneous gravitational lens parameter estimation and source reconstruction**, guided by **physics-informed learning**.

---

## 📌 Overview

PICoLT is a **Vision Transformer-based architecture** designed for analyzing **strong gravitational lensing systems**. Unlike traditional methods, it performs:

- 🔹 Lens parameter prediction  
- 🔹 Source galaxy reconstruction  
- 🔹 Physics consistency validation  

—all **within a single unified model**.

This project introduces a **differentiable gravitational lensing simulator** inside the training loop, ensuring that predictions are not only accurate but also **physically valid**.

---

## 🧠 Motivation

Upcoming surveys like the **LSST (Vera Rubin Observatory)** will detect **tens of thousands of strong lens systems**.

 Existing challenges:
- CNNs capture **local features only**
- Tasks are **handled separately**
- Lack of **physical consistency**

 PICoLT solves this by:
- Using **Vision Transformers (global understanding)**
- Performing **joint learning**
- Embedding **physics constraints directly into training**

---

## 🏗️ Architecture

<img width="337" height="603" alt="image" src="https://github.com/user-attachments/assets/3fac669a-d8c8-455f-8934-253975ad72b9" />

---

## ⚙️ Key Components

### 🔹 1. Vision Transformer Encoder
- Patch-based encoding (16×16)
- Multi-head self-attention
- Captures **long-range dependencies** (critical for lensing arcs)

---

### 🔹 2. Dual Decoders
- **MLP Head** → Lens parameter regression  
- **CNN Decoder** → Source image reconstruction  

---

### 🔹 3. Physics-Informed Simulator
- Implements gravitational lens equation and creates the input image 
- Uses:
  - SIE (Singular Isothermal Ellipsoid)
  - External shear
  - Ray-shooting
  - Bilinear interpolation

Ensures:
> 🧪 *Model predictions obey real astrophysical laws* by comparing reconstructed input image
> against original input image
> 
---

### 🔹 4. Hybrid Loss Function

Total Loss: L_total= λ_param L_param+ λ_src L_src+ λ_phys L_phys
- **Parameter Loss** → MSE  
- **Source Loss** → MAE + SSIM  
- **Physics Loss** → MSE based on Re-lensed vs observed image  

---

### 🔹 5. Dynamic Loss Weighting

Adaptive weighting strategy:
- Ensures balanced learning (no dominant task) across:
  - Accuracy
  - Reconstruction
  - Physics consistency
---

## 📊 Dataset

- 📦 **60,000 simulated samples**
- 🧪 Generated using:
  - SIE lens model
  - Sérsic galaxy profiles
  - PSF convolution
  - Realistic noise (HSC-like)

### Features:
- Single / Double / Irregular galaxies
- Realistic magnification (1.5 → 30)
- Data augmentation (rotations, flips)

---

