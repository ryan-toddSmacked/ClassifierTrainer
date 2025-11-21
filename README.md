# ChipTrainer - PyTorch Image Classification GUI

A comprehensive desktop application for training deep learning image classification models with PyTorch. Features a user-friendly interface with advanced training options, multi-metric tracking, model checkpointing, and GPU monitoring.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)

## ✨ Features

### Core Functionality
- **30+ Pre-trained Models**: ResNet, VGG, DenseNet, EfficientNet, MobileNet, and more from torchvision
- **Flexible Data Handling**: Train/validation/test split configuration with data subsetting
- **Real-time Monitoring**: Live training logs, progress tracking, and GPU statistics
- **Model Management**: Load pre-trained models, auto-select best from checkpoints, export trained models

### Advanced Training Options
- **Multiple Optimizers**: Adam, AdamW, SGD (with momentum), RMSprop, Adagrad
- **Learning Rate Schedulers**: StepLR, ReduceLROnPlateau, CosineAnnealingLR
- **Loss Functions**: CrossEntropyLoss, NLLLoss, BCELoss
- **Data Augmentation**: Random horizontal flip, random rotation
- **Early Stopping**: Configurable patience for validation improvement

### Evaluation & Metrics
- **12+ Metrics**: Accuracy, Precision, Recall, F-Beta Score (configurable beta), AUROC, AveragePrecision, Specificity, MatthewsCorrCoef, CohenKappa, JaccardIndex, Dice, HammingDistance
- **Multi-Metric Tracking**: Track multiple metrics simultaneously during training
- **Confusion Matrix**: Automatic generation on test set
- **Visualization**: Export matplotlib plots for all tracked metrics

### Smart Checkpointing
- **Automatic Saving**: Save models when validation metric improves by threshold
- **Best Model Selection**: Automatically load the best model from a directory
- **Descriptive Naming**: Models saved with architecture, metric, score, and timestamp

## 🚀 Quick Start

### Automated Setup

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Verify Python 3.10+ installation
- Create a virtual environment
- Detect NVIDIA GPU and install appropriate PyTorch version (CUDA or CPU)
- Install all dependencies
- Verify the installation

### Manual Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the environment:**
   - Windows: `.\.venv\Scripts\Activate.ps1`
   - Linux/Mac: `source .venv/bin/activate`

3. **Install PyTorch:**
   Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the appropriate command for your system.

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Starting the Application

```bash
python chip_trainer_gui.py
```

Or if using the virtual environment:
- Windows: `.\.venv\Scripts\python.exe chip_trainer_gui.py`
- Linux/Mac: `./.venv/bin/python chip_trainer_gui.py`

### Preparing Your Dataset

Organize images in the `ImageFolder` format:
```
/path/to/dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── class_3/
    └── ...
```

### Training Workflow

1. **Select Data Directory**: Choose your dataset folder
2. **Configure Model**: 
   - Select from 30+ pre-trained architectures
   - Optionally load a checkpoint to continue training
3. **Set Data Split**: Configure train/validation/test proportions (e.g., 0.7/0.2/0.1)
4. **Adjust Hyperparameters**:
   - Epochs, learning rate, batch size
   - Data subset (use portion of dataset or "all")
5. **Configure Advanced Options** (optional):
   - Optimizer and weight decay
   - Learning rate scheduler
   - Loss function
   - Data augmentation
   - Evaluation metric for model selection
   - Checkpointing settings
   - Multi-metric tracking
6. **Start Training**: Monitor progress in real-time
7. **Export Results**:
   - Export trained model with auto-generated filename
   - Export metric history (JSON + matplotlib plots)
   - View confusion matrix

## 🎯 Advanced Features

### Multi-Metric Tracking

Track multiple evaluation metrics during training without affecting model selection:

1. Open **Advanced Options**
2. Enable **Multi-Metric Tracking**
3. Set tracking frequency (every N epochs)
4. Select metrics to track (e.g., Accuracy, Precision, AUROC, F1-Score)
5. Configure F-Score beta if needed
6. After training, click **Export Metrics** to save:
   - JSON file with all metric values
   - Individual plots for each metric (train vs validation)
   - Confusion matrix heatmap (if test set available)

### Model Checkpointing

Automatically save models during training:

1. Enable checkpointing in **Advanced Options**
2. Set minimum improvement threshold (e.g., 0.5%)
3. Choose save directory
4. Models are saved as: `ModelName_epochN_metricScore_timestamp.pth`

### Loading Best Model

Auto-select the best model from a checkpoint directory:

1. Click **Load Best from Directory**
2. Select folder containing checkpoint files
3. System parses filenames and loads the highest-scoring model

## 📊 Supported Metrics

- **Classification**: Accuracy, Precision, Recall
- **F-Score Variants**: F1, F2, F0.5 (configurable beta)
- **Probabilistic**: AUROC, Average Precision
- **Specialized**: Specificity, Matthews Correlation Coefficient, Cohen's Kappa
- **Segmentation-style**: Jaccard Index, Dice Score
- **Distance**: Hamming Distance

## 🖥️ GPU Support

ChipTrainer automatically detects and uses NVIDIA GPUs if available:
- Real-time GPU monitoring (utilization, memory, temperature)
- Automatic CUDA version detection during setup
- Falls back to CPU if no GPU detected

## 📁 Project Structure

```
ChipTrainer/
├── chip_trainer_gui.py      # Main GUI application
├── bulk_trainer.py           # CLI batch training script
├── setup.ps1                 # Windows setup script
├── setup.sh                  # Linux/Mac setup script
├── verify_install.py         # Installation verification
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Requirements

- Python 3.10 or newer
- PyTorch 2.0+
- PyQt5
- torchvision
- torchmetrics
- GPUtil (for GPU monitoring)
- matplotlib (for metric visualization)
- Other dependencies in `requirements.txt`

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or request features via issues
- Submit pull requests with improvements
- Share your training results and use cases

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- Uses pre-trained models from [torchvision](https://pytorch.org/vision/stable/index.html)
- Metrics powered by [TorchMetrics](https://torchmetrics.readthedocs.io/)

## 💡 Tips

- **Start small**: Test with a small data subset before full training
- **Monitor GPU**: Keep an eye on memory usage to avoid OOM errors
- **Experiment**: Try different architectures and hyperparameters
- **Track metrics**: Enable multi-metric tracking to understand model behavior
- **Use checkpoints**: Save models regularly to avoid losing progress
- **Validate splits**: Ensure your data splits sum to 1.0 before training

## 🐛 Troubleshooting

**CUDA out of memory:**
- Reduce batch size
- Use a smaller model architecture
- Reduce input image size

**Training too slow:**
- Increase batch size (if GPU memory allows)
- Use data subset for testing
- Enable GPU acceleration

**Poor model performance:**
- Try different optimizers and learning rates
- Enable data augmentation
- Use appropriate evaluation metrics
- Check class balance in dataset

For more help, check the logs in the application window or run `python verify_install.py` to check your setup.


