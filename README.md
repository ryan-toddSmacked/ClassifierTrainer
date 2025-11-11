# ChipTrainer - Neural Network Image Classifier

Train a neural network with PyTorch using a simple Qt5 GUI interface.

## Overview

Designed to load images from subfolders whose folder name is the label of those images:
```
basedir/bananas/img1.png
basedir/bananas/img2.jpg
basedir/apples/img3.png
...
```

So, every subfolder in basedir is considered to be a label to be trained in a multiclass classifier.

## Features

- **Simple GUI Interface**: Easy-to-use Qt5 interface for training neural networks
- **Multiple Model Architectures**: 14 pre-trained models (ResNet, VGG, DenseNet, MobileNet, EfficientNet, etc.)
- **Automatic Class Detection**: Automatically detects classes from subfolder names
- **Real-time Training Progress**: Live updates showing training progress, loss, and accuracy
- **Configurable Parameters**: Adjust epochs, learning rate, batch size, optimizers, schedulers, and more
- **Custom Data Augmentation**: Configure individual augmentation transforms with custom parameters
- **GPU Support**: Automatically uses CUDA if available with specific GPU detection
- **Model Export**: Export models in multiple formats (PyTorch .pth, TorchScript .pt, ONNX .onnx)
- **Statistics Export**: Export comprehensive training statistics and visualizations
- **Configuration Save/Load**: Save and load training configurations as JSON

## Quick Setup

### Windows

Run the automated setup script:
```batch
setup.bat
```

This will:
- Check Python version (3.10+ required)
- Create a virtual environment
- Detect CUDA GPU and version
- Install PyTorch with appropriate CUDA support
- Install all dependencies

### Linux

Run the automated setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check Python version (3.10+ required)
- Create a virtual environment
- Detect CUDA GPU and version
- Install PyTorch with appropriate CUDA support
- Install all dependencies

## Manual Installation

If you prefer to install manually:

1. Ensure Python 3.10 or higher is installed
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux: `source .venv/bin/activate`

4. Install PyTorch (visit https://pytorch.org for the command matching your system)

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using the GUI

1. Activate the virtual environment (if not already activated):
   - Windows: `.venv\Scripts\activate`
   - Linux: `source .venv/bin/activate`

2. Run the application:
   ```bash
   python chip_trainer_gui.py
   ```

3. Select your data directory containing the image subfolders
4. Configure training parameters:
   - **Model Architecture**: Choose from 14 pre-trained models
   - **Epochs**: Number of training iterations (default: 10)
   - **Learning Rate**: How fast the model learns (default: 0.001)
   - **Batch Size**: Number of images processed together (default: 32)
4. Click "Start Training" to begin
5. Monitor progress in the log area
6. The trained model will be saved as `trained_model.pth` in the parent directory of your data folder

### Directory Structure Example

```
my_dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.png
│   └── cat3.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.png
│   └── dog3.jpg
└── birds/
    ├── bird1.jpg
    ├── bird2.png
    └── bird3.jpg
```

## Requirements

- Python 3.10+
- PyQt5
- PyTorch
- torchvision
- Pillow
- numpy

## Technical Details

- Uses ResNet-18 as the base model with transfer learning
- Supports common image formats (JPEG, PNG, etc.)
- Images are automatically resized to 224x224 pixels
- Uses Adam optimizer
- Cross-entropy loss for multiclass classification



