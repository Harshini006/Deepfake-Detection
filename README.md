# End-to-End Deepfake Detection Engine from Scratch

A structured Computer Vision pipeline built using **Python, OpenCV, and TensorFlow** designed to ingest raw, unstandardized facial media, isolate spatial regions of interest, and optimize inputs for Deep Learning systems.

## 🏗️ Current Status: Phase 1 (Data Engineering Complete)
The production-grade data ingestion and preprocessing pipeline is fully implemented. The core Convolutional Neural Network (CNN) backbone and evaluation metrics will be integrated in the upcoming development sprint.

### 🔄 Implemented Pipeline Workflow
1. **Automated Structural Sanitization (`clean_dataset.py`):** Parses raw image directories, utilizes PIL integrity checks to instantly drop corrupted file byte-streams, isolates human facial structures via OpenCV Haar Cascades with a 15% padding margin to preserve boundary artifacts, and standardizes all outputs.
2. **Online Tensor Preprocessing (`data_pipeline.py`):** Generates optimized `tf.data.Dataset` streams with memory prefetching, scales pixel arrays dynamically to `[0, 1]`, and applies runtime spatial augmentations (horizontal flips and rotations) to prevent model overfitting.

## 📅 Roadmap & Next Steps
- [x] Phase 1: Messy dataset ingestion and automated face cropping pipeline.
- [ ] Phase 2: Design Transfer Learning model core using an EfficientNetB0 backbone.
- [ ] Phase 3: Implement binary cross-entropy optimization and training tracking loops.
- [ ] Phase 4: Construct visual diagnostics (Confusion Matrix & Accuracy/Loss Curves).