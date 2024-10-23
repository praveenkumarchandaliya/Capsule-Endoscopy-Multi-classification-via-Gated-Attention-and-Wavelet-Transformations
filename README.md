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
