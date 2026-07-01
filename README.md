# Real-Time Deepfake Detection System via MobileNetV2 Transfer Learning

An end-to-end, high-performance computer vision pipeline designed to detect digital facial forgeries and deepfakes. This system transitions from a baseline scratch-built CNN (MesoNet-4) to an optimized **MobileNetV2 Transfer Learning architecture**, specifically calibrated to handle real-world webcam feeds, indoor lighting fluctuations, and high-fidelity generative AI artifacts.

---

## 🚀 Key Features & Architectural Upgrades

* **Feature Extraction Backbone:** Upgraded from a shallow 4-layer custom CNN to a pre-trained **MobileNetV2** engine (trained on ImageNet), providing deep spatial texture awareness and preventing weight collapse.
* **Live Multi-Face Tracking:** Integrates an OpenCV Haar Cascade frontier pipeline to dynamically track, crop, and pad individual regions of interest (ROIs) from any arbitrary image frame size.
* **Strict Regularization & Data Augmentation:** Implements real-time spatial transformations (random flips, rotations, and zooms) alongside a `Dropout(0.4)` layer to eliminate background memorization and overfitting.
* **Inference Calibration Layer:** Deploys a customized dynamic boundary scaling system centered around a calibrated threshold ($0.582$) to balance fine-grained webcam noise against high-fidelity AI-generated portraits.

---

## 📊 System Metrics & Training Logs

The pipeline was trained using a localized image subset accelerated by a Google Colab Tesla T4 GPU:

* **Optimizer:** Adam ($\alpha = 0.001$)
* **Loss Function:** Binary Crossentropy
* **Training Epochs:** 10
* **Validation Accuracy Achieved:** **88.63%**
* **Model Convergence Stability:** The validation loss smoothly aligned with training loss curves at $\approx 0.297$, proving excellent generalization on unseen faces.

---

## ⚙️ Repository Structure

```text
├── DEEPFAKE_DETECTION.ipynb     <- Fully documented, executed training & inference pipeline
├── src/
│   └── predict_inference.py     <- Calibrated production inference script
├── .gitignore                   <- Bypasses heavy dataset zips and local checkpoints
└── README.md                    <- Production documentation
