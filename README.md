# Image Categorization Project

A comprehensive deep learning project for image classification using PyTorch, supporting both DNN and CNN architectures on the CINIC-10 dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Evaluation](#evaluation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a complete image classification pipeline using PyTorch, designed to classify images from the CINIC-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The project supports multiple neural network architectures and provides comprehensive evaluation and analysis tools.

## Features

- **Multiple Model Architectures**: DNN (Vanilla, Deep) and CNN (Simple, Advanced)
- **Flexible Training**: Full dataset training or quick test mode with subset data
- **Comprehensive Evaluation**: Top-1 accuracy, F1-score, precision, recall metrics
- **Automatic Evaluation**: Integrated evaluation pipeline with prediction visualization
- **Exploratory Data Analysis**: Independent EDA script with beautiful visualizations
- **GPU Support**: CUDA-enabled training and inference
- **Progress Tracking**: Real-time training and validation progress monitoring
- **Checkpoint Management**: Automatic model saving and loading
- **Statistical Analysis**: Multi-run evaluation with mean and standard deviation

## Project Structure

```
image-categorization-prosit/
├── main.py                          # Main training script
├── model_architecture/
│   ├── DNN.py                       # DNN model definitions
│   └── CNN.py                       # CNN model definitions
├── scripts/
│   ├── evaluate.py                  # Model evaluation script
│   └── eda.py                       # Exploratory data analysis script
├── checkpoints/                     # Saved model checkpoints
├── experiment_results/              # Evaluation results and visualizations
├── dataset/                         # CINIC-10 dataset directory
│       ├── train/
│       ├── valid/
│       └── test/
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM recommended
- 10GB+ free disk space for dataset

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib seaborn scikit-learn pandas numpy tqdm pillow
```

### Dataset Setup

1. Download the CINIC-10 dataset
2. Extract to `./dataset/` directory
3. Ensure the following structure:
   ```
   dataset/
   ├── train/
   │   ├── airplane/
   │   ├── automobile/
   │   └── ...
   ├── valid/
   │   ├── airplane/
   │   ├── automobile/
   │   └── ...
   └── test/
       ├── airplane/
       ├── automobile/
       └── ...
   ```

## Quick Start

### Quick Test Mode (Recommended for first run)

Test the pipeline with a small subset of data:

```bash
python main.py
```

This runs in quick test mode by default with 4 samples per class.

### Full Training

Train a model on the complete dataset:

```bash
python main.py --model_type cnn --model_variant advanced --auto_evaluate
```

## Usage

### Command Line Interface

The main script supports various command-line arguments:

```bash
python main.py [OPTIONS]
```

#### Options

- `--model_type {dnn,cnn}`: Model architecture type (default: dnn)
- `--model_variant {vanilla,deep,simple,advanced}`: Model variant (default: vanilla)
- `--test_mode`: Enable quick test mode with subset data
- `--test_samples N`: Number of samples per class in test mode (default: 4)
- `--data_root PATH`: Dataset root directory (default: ./dataset)
- `--auto_evaluate`: Run automatic evaluation after training
- `--num_eval_runs N`: Number of evaluation runs (default: 3)

#### Examples

**Quick test with DNN:**
```bash
python main.py --model_type dnn --model_variant vanilla --test_mode
```

**Full CNN training with evaluation:**
```bash
python main.py --model_type cnn --model_variant advanced --auto_evaluate
```

**Custom test samples:**
```bash
python main.py --test_mode --test_samples 10 --model_type cnn --model_variant simple
```

### Quick Configuration

For easy testing without command-line arguments, modify the configuration section in `main.py`:

```python
# Quick Configuration - Change these values for easy testing
QUICK_TEST_MODE = True          # Set to True for quick testing
QUICK_TEST_SAMPLES = 4          # Number of samples for quick test
QUICK_MODEL_TYPE = 'cnn'        # 'dnn' or 'cnn'
QUICK_MODEL_VARIANT = 'simple'  # 'vanilla', 'deep', 'simple', 'advanced'
```

## Model Architectures

### DNN Models

#### Vanilla DNN
- **Layers**: 3 Linear layers
- **Activation**: ReLU
- **Regularization**: Dropout (0.5)
- **Parameters**: ~1.2M

#### Deep DNN
- **Layers**: 5 Linear layers with BatchNorm
- **Activation**: ReLU
- **Regularization**: Dropout (0.3-0.5)
- **Parameters**: ~2.1M

### CNN Models

#### Simple CNN
- **Layers**: 3 Conv2d + 3 Linear layers
- **Pooling**: MaxPool2d
- **Activation**: ReLU
- **Regularization**: Dropout (0.5)
- **Parameters**: ~1.8M

#### Advanced CNN
- **Layers**: 6 Conv2d + 2 Linear layers
- **Pooling**: MaxPool2d + AdaptiveAvgPool2d
- **Normalization**: BatchNorm2d
- **Activation**: ReLU
- **Regularization**: Dropout (0.5)
- **Parameters**: ~2.3M

## Evaluation

### Automatic Evaluation

Evaluation runs automatically after training when `--auto_evaluate` is enabled:

```bash
python main.py --model_type cnn --model_variant advanced --auto_evaluate
```

### Manual Evaluation

Run detailed evaluation on a trained model:

```bash
python scripts/evaluate.py --model_path checkpoints/cnn_advanced_trained.pth --model_type cnn --model_variant advanced --detailed
```

### Statistical Evaluation

Run multiple evaluation runs for statistical analysis:

```bash
python scripts/evaluate.py --model_path checkpoints/dnn_vanilla_trained.pth --model_type dnn --model_variant vanilla --num_runs 5
```

#### Evaluation Metrics

- **Top-1 Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall (weighted average)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Confusion Matrix**: Detailed class-wise performance analysis

### Evaluation Outputs

- **Results JSON**: Numerical metrics saved to `experiment_results/`
- **Confusion Matrix**: Visual performance analysis
- **Prediction Visualization**: Sample predictions with actual vs predicted labels
- **Detailed Report**: Comprehensive classification report

## Exploratory Data Analysis

Run independent EDA on the CINIC-10 dataset:

```bash
python scripts/eda.py
```

### EDA Outputs

- **Class Distribution**: Analysis across train/valid/test splits
- **Dataset Structure**: Split proportions and balance analysis
- **Comprehensive Analysis**: Single visualization with all distribution charts

## Configuration

### Training Parameters

- **Epochs**: 30 (full training), 5 (test mode)
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Validation**: 10% of training data

### Data Augmentation

- **Resize**: 224x224 pixels
- **Random Horizontal Flip**: 50% probability
- **Normalization**: ImageNet mean and std

### Hardware Requirements

- **GPU**: CUDA-compatible (recommended)
- **RAM**: 8GB+ (16GB+ for full dataset)
- **Storage**: 10GB+ for dataset and checkpoints

## Results

### Expected Performance

| Model | Architecture | Test Accuracy | F1-Score |
|-------|-------------|---------------|----------|
| DNN | Vanilla | ~45-55% | ~0.45-0.55 |
| DNN | Deep | ~50-60% | ~0.50-0.60 |
| CNN | Simple | ~60-70% | ~0.60-0.70 |
| CNN | Advanced | ~65-75% | ~0.65-0.75 |

### Common Issues

- **Vehicle Confusion**: Cars vs trucks (most common misclassification)
- **Animal Confusion**: Cats vs dogs (similar features)
- **Overfitting**: Monitor training vs validation accuracy

## Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Dataset not found**
   ```
   FileNotFoundError: Expected subfolder not found: ./dataset/CINIC-10
   ```
   - Ensure dataset is in correct location
   - Check directory structure matches requirements

3. **Out of memory**
   - Reduce batch size in `main.py`
   - Use test mode for initial testing
   - Close other applications

4. **Model stuck during training**
   - Check validation batch limit (set to 10 for efficiency)
   - Monitor progress output

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed
- **Data Loading**: Adjust `num_workers` based on CPU cores
- **Memory**: Use `pin_memory=True` for faster GPU transfer

## Acknowledgments

- Mr. Dennis Owusu Asamoah for the wonderful Prosit
- CINIC-10 dataset creators
- PyTorch team for the deep learning framework
- Contributors and testers