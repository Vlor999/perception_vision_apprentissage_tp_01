# Neural Network Object Detector

A PyTorch-based image classification and bounding box regression project. This project implements multiple neural network architectures to detect and localize objects in images across three categories: **motorcycles**, **airplanes**, and **faces**.

## Features

- **Multiple architectures**: SimpleDetector, DeeperDetector, VGG-inspired, and ResNet-based models
- **Dual-task learning**: Classification + Bounding box regression
- **Transfer learning**: Pre-trained ResNet18 with optional feature freezing
- **Cross-platform**: Supports CUDA, Apple MPS, and CPU

## Project Structure

```
├── train.py              # Main training script
├── eval.py               # Model evaluation script
├── predict.py            # Prediction visualization script
├── compare_models.py     # Compare all architectures
├── src/
│   ├── config.py         # Configuration and hyperparameters
│   ├── dataset.py        # PyTorch Dataset implementation
│   ├── network.py        # Neural network architectures
│   └── arguments/        # CLI argument handling
├── output/               # Saved models and training outputs
└── matieres/5MMVORF/01-dataset/  # Dataset location
```

## Requirements

- Python >= 3.13
- CUDA GPU (recommended), Apple Silicon (MPS), or CPU

## Installation

### 1. Install uv (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Clone and setup the project

```bash
git clone <repository-url>
cd 01-intro-to-neural-nets
```

### 3. Install dependencies with uv

```bash
# Create virtual environment and install all dependencies
uv sync
```

## Dataset Setup

The dataset should be placed in `matieres/5MMVORF/01-dataset/` with the following structure:

```
matieres/5MMVORF/01-dataset/
├── images/
│   ├── airplane/
│   ├── face/
│   └── motorcycle/
└── annotations/
    ├── airplane.csv
    ├── face.csv
    └── motorcycle.csv
```

## Usage

### Training a Model

```bash
# Train with default settings (SimpleDetector, 20 epochs)
uv run python train.py

# Train with a specific model
uv run python train.py --model simple          # SimpleDetector
uv run python train.py --model deeper          # DeeperDetector
uv run python train.py --model vgg_inspired    # VGG-inspired architecture
uv run python train.py --model resnet          # ResNet18 (frozen features)
uv run python train.py --model resnet_unfrozen # ResNet18 (trainable features)

# Customize training parameters
uv run python train.py --model resnet --epoch-size 50 --batch-size 16

# Force CPU usage
uv run python train.py --cpu-only

# Disable model/plot saving
uv run python train.py --save-model false --save-plots false
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `simple` | Model architecture (`simple`, `deeper`, `vgg_inspired`, `resnet`, `resnet_unfrozen`) |
| `--epoch-size` | `20` | Number of training epochs |
| `--batch-size` | `32` | Batch size (auto-adjusted for MPS) |
| `--workers` | auto | Number of data loader workers |
| `--cpu-only` | `false` | Force CPU usage |
| `--save-model` | `true` | Save trained model |
| `--save-plots` | `true` | Save training plots |

### Evaluating a Model

```bash
# Evaluate the best model
uv run python eval.py output/best_model.pth

# Evaluate the last model
uv run python eval.py output/last_model.pth
```

### Making Predictions

```bash
# Predict on test dataset
uv run python predict.py --directory output/test_data.csv

# Predict on a single image
uv run python predict.py --filename path/to/image.jpg

# Show only misclassified images
uv run python predict.py --directory output/test_data.csv --show-all-images false

# Use the last model instead of best
uv run python predict.py --directory output/test_data.csv --model last

# Save prediction result
uv run python predict.py --filename image.jpg --save-file true --output-file result.jpg
```

### Comparing All Models

```bash
# Train and compare all architectures
uv run python compare_models.py
```

## Output Files

After training, the following files are generated in `output/`:

| File | Description |
|------|-------------|
| `best_model.pth` | Model with best validation accuracy |
| `last_model.pth` | Model from the last epoch |
| `convergence_plot.png` | Training/validation loss and accuracy curves |
| `training_data.csv` | Training split file paths |
| `val_data.csv` | Validation split file paths |
| `test_data.csv` | Test split file paths |

## Model Architectures

| Model | Description | Parameters |
|-------|-------------|------------|
| `SimpleDetector` | Basic CNN with 2 conv blocks | ~2M |
| `DeeperDetector` | Deeper CNN with 4 conv blocks | ~5M |
| `VGGInspired` | VGG11-inspired architecture | ~10M |
| `ResnetObjectDetector` | Pre-trained ResNet18 backbone | ~11M |

## Configuration

Key configuration options in `src/config.py`:

```python
NUM_EPOCHS = 20        # Training epochs
INIT_LR = 1e-4         # Learning rate
BATCH_SIZE = 32        # Batch size (16 for MPS)
LABELW = 1.0           # Classification loss weight
BBOXW = 1.0            # Bounding box loss weight
```

## Device Support

The project automatically detects and uses the best available device:

1. **CUDA** - NVIDIA GPUs (fastest)
2. **MPS** - Apple Silicon GPUs (M1/M2/M3)
3. **CPU** - Fallback option

## License

This project was developed for educational purposes at ENSIMAG/Grenoble INP.
