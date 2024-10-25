# Capsule-Endoscopy-Multi-classification-via-Gated-Attention-and-Wavelet-Transformations
This repository implements a novel deep learning model for automatic classification of gastrointestinal abnormalities in VCE frames. It integrates Omni-Dimensional Gated Attention and wavelet transformation into a modified ResNet18-based deep encoder-decoder architecture, enabling intricate multi-level feature extraction.

## Getting Started
To get started with this project, clone the repository and install the required packages. Follow the instructions below for setup.

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/09Srinivas2005/Capsule-Endoscopy-Multi-classification-via-Gated-Attention-and-Wavelet-Transformations.git

2. **Navigate to the Project Directory**
   ```bash
   cd Capsule-Endoscopy-Multi-classification-via-Gated-Attention-and-Wavelet-Transformations

3. **Install Required Packages**
   Make sure you have Python installed. Then, run the following command to install the necessary packages:
   ```bash
   pip install -r requirements.txt
## Usage
1. To run the training script, use the following command:
   ```bash
   python Training.py

2. For testing the model, execute:
   ```bash
   python Testing.py
## Checkpoint Files
   The model checkpoints can be found in the root directory. The checkpoint file "CP-PT_50_epochs.pth" contains the model weights after training for 50 epochs.

## Results

The performance of the proposed model was evaluated on Datasets provided by Capsule Vision 2024 Challenge and compared against two baseline models, VGG16 and ResNet50. The model demonstrated superior performance across multiple metrics, highlighting its capability to accurately classify gastrointestinal abnormalities in Video Capsule Endoscopy (VCE) frames.

### Key Metrics:

- **AUC (Area Under the Curve)**: 87.49
- **Balanced Accuracy**: 94.81
- **Accuracy**: 91.19
- **F1 Score**: 91.11
- **Precision**: 91.17
- **Recall**: 91.19
- **Specificity**: 98.44

The table below provides a comparison of the proposed model's performance with the baseline models:

| Model          | AUC   | Balanced Accuracy | Accuracy | F1 Score | Precision | Recall | Specificity |
|----------------|-------|-------------------|----------|----------|-----------|--------|-------------|
| **VGG16**      | 91.61 |       56.84       |  69.06   |   48.44  |   52.46   |  54.30 |    96.97    |
| **ResNet50**   | 87.10 |         -         |  76.02   |   76.0   |   78.0    |  76.0  |      -      |
| **Proposed**   | 87.49 |       94.81       |  91.19   |   91.11  |   91.17   |  91.19 |    98.44    |

The superior results achieved by the proposed model shows its effectiveness in identifying and classifying gastrointestinal abnormalities accurately.

