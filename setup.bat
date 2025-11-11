@echo off
REM ChipTrainer Setup Script for Windows
REM This script sets up the Python environment and installs PyTorch with appropriate CUDA support

echo ============================================
echo ChipTrainer Setup Script
echo ============================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

REM Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check if Python version is 3.10 or higher
if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.10 or higher is required
    echo Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 10 (
    echo ERROR: Python 3.10 or higher is required
    echo Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)

echo Python version check passed!
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected or nvidia-smi not found
    echo Installing PyTorch with CPU support only
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo NVIDIA GPU detected
    echo.
    
    REM Get CUDA version from nvidia-smi
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv^,noheader 2^>nul') do set DRIVER_VERSION=%%i
    echo NVIDIA Driver Version: %DRIVER_VERSION%
    
    REM Check CUDA toolkit version
    echo Checking CUDA Toolkit version
    nvcc --version >nul 2>&1
    if errorlevel 1 (
        echo CUDA Toolkit not found via nvcc, checking nvidia-smi
        
        REM Try to get CUDA version from nvidia-smi
        for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set CUDA_LINE=%%i
        if defined CUDA_LINE (
            for /f "tokens=9" %%j in ("%CUDA_LINE%") do set CUDA_VERSION=%%j
            echo CUDA Version from nvidia-smi: %CUDA_VERSION%
        ) else (
            echo Could not detect CUDA version, defaulting to CUDA 12.6
            set CUDA_VERSION=12.6
        )
    ) else (
        REM Get CUDA version from nvcc
        for /f "tokens=5" %%i in ('nvcc --version ^| findstr /C:"release"') do set CUDA_VERSION=%%i
        set CUDA_VERSION=%CUDA_VERSION:~0,-1%
        echo CUDA Toolkit Version: %CUDA_VERSION%
    )
    
    REM Determine PyTorch CUDA variant based on CUDA version
    echo.
    echo Determining appropriate PyTorch version
    echo Supported CUDA versions: 12.6, 12.8, 13.0
    
    REM Extract major and minor CUDA version
    for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION%") do (
        set CUDA_MAJOR=%%a
        set CUDA_MINOR=%%b
    )
    
    if "%CUDA_MAJOR%"=="13" (
        echo Installing PyTorch with CUDA 13.0 support
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ) else if "%CUDA_MAJOR%"=="12" (
        if "%CUDA_MINOR%" GEQ "8" (
            echo Installing PyTorch with CUDA 12.8 support
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        ) else if "%CUDA_MINOR%" GEQ "6" (
            echo Installing PyTorch with CUDA 12.6 support
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        ) else (
            echo CUDA 12.%CUDA_MINOR% detected
            echo PyTorch supports CUDA 12.6, 12.8, and 13.0
            echo Installing PyTorch with CUDA 12.6 support closest match
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        )
    ) else (
        echo CUDA version %CUDA_VERSION% detected
        echo PyTorch currently supports CUDA 12.6, 12.8, and 13.0
        echo Installing PyTorch with CUDA 13.0 support latest
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    )
)

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo.

REM Install other requirements
echo Installing additional requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo.

echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To use ChipTrainer:
echo 1. Activate the virtual environment:
echo    .venv\Scripts\activate
echo.
echo 2. Run the GUI:
echo    python chip_trainer_gui.py
echo.
echo Verifying PyTorch installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo.
pause
