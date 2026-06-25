# End-to-End Deepfake Detection Engine from Scratch

A structured Computer Vision pipeline built using **Python, OpenCV, and TensorFlow** designed to ingest raw, unstandardized facial media, isolate spatial regions of interest, and optimize inputs for Deep Learning systems.

🚀 **Current Status: Phase 2 (Model Optimization Active)**

The production-grade data ingestion pipeline and baseline Convolutional Neural Network (CNN) architecture are fully implemented. The system is currently undergoing training loops and regularization tuning to improve generalization.

## ⚙️ Implemented Pipeline Workflow

1. **Automated Structural Sanitization (`src/clean_dataset.py`):** Parses raw image directories, utilizes PIL integrity checks to instantly drop corrupted file byte-streams, isolates human facial structures via OpenCV Haar Cascades with a 15% padding margin to preserve boundary artifacts, and standardizes all outputs.
2. **Online Tensor Preprocessing (`notebook/Deepfake_Detection.ipynb`):** Generates optimized `tf.data.Dataset` streams with memory prefetching, scales pixel arrays dynamically to `[0, 1]`, and feeds streamlined batches directly into the tensor engine.
3. **Deep Learning Core & Diagnostics (`notebook/Deepfake_Detection.ipynb`):** Built a baseline Convolutional Neural Network (CNN) with custom Conv2D and MaxPooling layers optimized via high-speed cloud T4 GPU acceleration. Evaluated training performance curves using Matplotlib to identify variance and overfitting boundaries.

## ☁️ Execution Guide (Google Colab Deployment)

To bypass local hardware limitations and directory path glitches, execute the core model training notebook directly in the cloud:
1. Open Google Colab and import the `notebook/Deepfake_Detection.ipynb` from this repository.
2. Compress your local preprocessed folder into `dataset_clean.zip` and upload it to your Google Drive (`My Drive/deepfake dataset/`).
3. Connect your notebook to the cloud GPU: Go to **Runtime -> Change runtime type -> Hardware Accelerator -> T4 GPU**.
4. Run the code cells sequentially to automatically mount Drive, unpack image streams, and initiate model diagnostics.

## 🗺️ Roadmap & Next Steps

- [x] **Phase 1:** Messy dataset ingestion and automated face cropping pipeline.
- [ ] **Phase 2:** Design Transfer Learning model core using an Advanced Backbone / MesoNet.
- [x] **Phase 3:** Implement binary cross-entropy optimization and training tracking loops.
- [x] **Phase 4:** Construct visual diagnostics (Confusion Matrix & Accuracy/Loss Curves).
