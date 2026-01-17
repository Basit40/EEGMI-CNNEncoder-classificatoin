# Contrastive Learning for EEG Motor Imagery Classification

This repository implements a contrastive learning framework for EEG-based motor imagery classification using PyTorch. The model combines self-supervised pre-training with supervised fine-tuning to improve classification performance on EEG datasets.

## Features

- **Contrastive Pre-training**: Uses two different transformations on EEG data to learn robust representations
- **EEG-Specific Transformations**: Includes time warping, noise addition, filtering, and other EEG-specific augmentations
- **Multi-Subject Training**: Trains on multiple subjects and tests on held-out subjects
- **Euclidean Alignment**: Aligns EEG data across subjects for better generalization
- **Visualization Tools**: Includes t-SNE visualization and confusion matrix plotting

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- tqdm
- torchinfo
- warmup_scheduler

Install dependencies with:
```bash
pip install torch numpy scipy scikit-learn matplotlib tqdm torchinfo warmup_scheduler


Project Structure

├── contrastX.py          # Main training and evaluation script
├── NNN.py                # SSL network architecture
├── ContrastiveLoss0.py   # Contrastive loss implementation
├── ContrastiveLoss4.py   # Alternative contrastive loss
├── aligning_data.py      # Euclidean alignment functions
├── choice_random.py      # Index selection utilities
├── transform_rrr.py      # Data transformation/augmentation functions
├── T_SNE.py              # t-SNE visualization utilities
├── mit_utils.py          # MIT-BIH dataset utilities
└── logs/                 # Directory for saved models and results



Datasets:
The code supports two BCI Competition IV datasets:

BCI Competition IV 2b (Two-class)
Classes: Left hand (0) vs Right hand (1)

Channels: 3 EEG channels

Subjects: 9 subjects

BCI Competition IV 2a (Four-class)
Classes: Left hand, Right hand, Foot, Tongue

Channels: 22 EEG channels

Subjects: 9 subjects

Note: Update the file paths in the code to point to your local dataset directories.



Usage
Basic Usage:
bash
python contrastX.py -F1 <transformation1> -F2 <transformation2>
Example:
bash
python contrastX.py -F1 time_warp -F2 noise




Available Transformations:

time_warp: Time warping with random stretching/squeezing

noise: Additive noise

scale: Signal scaling

DC: DC offset addition

negate: Signal negation

hor_flip: Horizontal flipping

permute: Segment permutation

cutout_resize: Cutout with resize

cutout_zero: Cutout with zeros

crop_resize: Crop and resize

move_avg: Moving average filtering

lowpass: Low-pass filtering

highpass: High-pass filtering

bandpass: Band-pass filtering




Training Process:


Data Preparation:

Loads EEG data from 9 subjects

Splits data into training, validation, and test sets

Applies Euclidean alignment across subjects

Normalizes data (z-score normalization)


2. Contrastive Pre-training:

Applies two different transformations to each sample

Trains encoder using contrastive loss

Saves best encoder weights

3. Supervised Fine-tuning:

Loads pre-trained encoder

Adds classification head

Trains on labeled data with cross-entropy loss

Validates after each epoch

4. Testing:

Evaluates on held-out test subject

Generates predictions and visualizations

Computes accuracy, confusion matrix, and F1 scores



Key Parameters:

data_start: Start index for data slicing (default: 0)

data_length: Length of data segments (default: 550)

num_indices_for_V: Validation samples per class (default: 30)

num_indices_for_T: Test samples per class (default: 30)

batch_size: Batch size (default: 64)

epochs: Contrastive pre-training epochs (default: 10)

epochs_t: Supervised training epochs (default: 300)

learning_rate: Adam optimizer LR (default: 0.001)



Data Settings:
Target subject for testing is set to subject 9 by default (can be changed in code)

Training uses data from all other subjects

Random sampling ensures balanced class representation


Output
The script generates:


Saved Files:
Models: Best and latest weights in logs/ directory

Visualizations:

	Accuracy and loss plots

	t-SNE feature visualizations (true and predicted)

	Confusion matrices

Logs: Training statistics and test results

Console Output:
Training progress with loss and accuracy

Validation metrics during fine-tuning

Final test accuracy and F1 scores

Per-class performance metrics