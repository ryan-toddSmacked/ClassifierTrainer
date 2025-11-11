#!/bin/bash
# ChipTrainer Setup Script for Linux
# This script sets up the Python environment and installs PyTorch with appropriate CUDA support

set -e  # Exit on error

echo "============================================"
echo "ChipTrainer Setup Script"
echo "============================================"
echo ""

# Check if Python 3 is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10 or higher using your package manager"
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "  Fedora: sudo dnf install python3.10"
    echo "  Arch: sudo pacman -S python"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Found Python version: $PYTHON_VERSION"

# Extract major and minor version
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

# Check if Python version is 3.10 or higher
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Python version check passed!"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping creation"
else
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        echo "You may need to install python3-venv:"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
    echo "Virtual environment created successfully"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi &> /dev/null
    if [ $? -eq 0 ]; then
        echo "NVIDIA GPU detected!"
        echo ""
        
        # Get NVIDIA driver version
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        echo "NVIDIA Driver Version: $DRIVER_VERSION"
        
        # Check CUDA toolkit version
        echo "Checking CUDA Toolkit version..."
        if command -v nvcc &> /dev/null; then
            # Get CUDA version from nvcc
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
            echo "CUDA Toolkit Version: $CUDA_VERSION"
        else
            echo "CUDA Toolkit not found via nvcc, checking nvidia-smi..."
            
            # Try to get CUDA version from nvidia-smi
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
            if [ -z "$CUDA_VERSION" ]; then
                echo "Could not detect CUDA version, defaulting to CUDA 12.1"
                CUDA_VERSION="12.1"
            else
                echo "CUDA Version from nvidia-smi: $CUDA_VERSION"
            fi
        fi
        
        # Determine PyTorch CUDA variant based on CUDA version
        echo ""
        echo "Determining appropriate PyTorch version..."
        echo "Supported CUDA versions: 12.6, 12.8, 13.0"
        
        # Extract major and minor CUDA version
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        
        if [ "$CUDA_MAJOR" -eq 13 ]; then
            echo "Installing PyTorch with CUDA 13.0 support..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            if [ "$CUDA_MINOR" -ge 8 ]; then
                echo "Installing PyTorch with CUDA 12.8 support..."
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
            elif [ "$CUDA_MINOR" -ge 6 ]; then
                echo "Installing PyTorch with CUDA 12.6 support..."
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
            else
                echo "CUDA 12.$CUDA_MINOR detected"
                echo "PyTorch supports CUDA 12.6, 12.8, and 13.0"
                echo "Installing PyTorch with CUDA 12.6 support (closest match)..."
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
            fi
        else
            echo "CUDA version $CUDA_VERSION detected"
            echo "PyTorch currently supports CUDA 12.6, 12.8, and 13.0"
            echo "Installing PyTorch with CUDA 13.0 support (latest)..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
        fi
    else
        echo "nvidia-smi found but failed to run"
        echo "Installing PyTorch with CPU support only..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "No NVIDIA GPU detected or nvidia-smi not found"
    echo "Installing PyTorch with CPU support only..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi
echo ""

# Install other requirements
echo "Installing additional requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi
echo ""

echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "To use ChipTrainer:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run the GUI:"
echo "   python chip_trainer_gui.py"
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""
