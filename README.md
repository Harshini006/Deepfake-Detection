# Deepfake Detection System using MesoNet-4 Architecture

An end-to-end, high-performance computer vision pipeline designed to detect digital facial forgeries and deepfakes using specialized micro-texture analysis. This project transitions from a baseline convolutional neural network (CNN) regularized with dynamic data augmentation to a scratch-built **MesoNet-4** architecture optimized for real-time localized multi-face forgery detection.

---

## 📌 Project Overview & Roadmap Progress
- [x] **Phase 1: Data Engineering Pipeline** (`clean_dataset.py` with automated face cropping and spatial margins).
- [x] **Phase 2: Baseline Model Optimization** (Mitigated heavy overfitting using dynamic TensorFlow regularization layers).
- [x] **Phase 3: Specialized MesoNet-4 Engine** (Implemented micro-texture kernel analysis blocks from scratch).
- [x] **Phase 4: Diagnostics & Inference** (Deployed an automated, any-size multi-face localized testing pipeline).

---

## 🧠 Architectural Insights

### The Overfitting Breakthrough (Data Augmentation)
Our initial baseline CNN suffered from severe over-memorization (Training Accuracy at ~99%, while Validation Accuracy plateaued at a wide variance gap). To enforce generalized pattern tracking instead of raw pixel memorization, a dynamic regularization layer block was embedded:
* **Horizontal Flips:** Mirror image adjustments.
* **Random Rotation & Zoom (0.1):** Prevents static positional anchoring.

**Result:** The validation variance gap closed cleanly, locking in a highly stable **85.20% Validation Accuracy** on the baseline.

### Transition to MesoNet-4
Traditional deep architectures (VGG, ResNet) extract heavy macro-level facial structures, which modern deepfake generative tools (like FaceSwap or DeepFaceLab) replicate flawlessly. 

MesoNet-4 bypasses this by utilizing a lightweight footprint (4 compact Convolution blocks) engineered with specialized multi-scale kernel sizes ($3\times3$ and $5\times5$). Instead of looking at general facial geometry, it functions like a digital magnifying glass—targeting microscopic blending boundaries, compression noise, and skin texture inconsistencies left behind by artificial generation.

---

## 📊 Technical Performance & Metric Log
After training over 10 optimized epochs accelerated by a Google Colab **Tesla T4 GPU**, the final model statistics registered:

| Model Architecture | Training Accuracy | Validation Accuracy | Optimization Status |
| :--- | :---: | :---: | :--- |
| Baseline CNN (Standard) | ~99.00% | ~74.00% | Severe Overfitting ❌ |
| **Baseline + Data Augmentation** | **94.00%** | **85.20%** | Balanced Variance Matrix  |
| **MesoNet-4 (From Scratch)** | **91.80%** | **84.75%** | High Generalization Index 🎯 |

---

## 📂 Repository Structure
```text
├── dataset_clean/             # Processed, face-isolated image assets
│   ├── train/                 # Binary classified folders (Real / Fake)
│   └── validation/            # Evaluation image sets
├── notebook/
│   └── Deepfake_Detection.ipynb  # End-to-end training, curves, and validation notebook
├── best_mesonet_deepfake.keras  # Exported high-performance production model weights
└── README.md                  # System documentation
