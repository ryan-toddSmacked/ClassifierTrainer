# setup.ps1 - Prepares the Python environment for ChipTrainer

# --- Configuration ---
$RequiredPythonVersion = "3.10"
$VenvDir = ".venv"
# Supported CUDA versions and their corresponding PyTorch wheel identifiers
$SupportedCudaVersions = @{
    "12.6" = "cu126";
    "12.8" = "cu128";
    "13.0" = "cu130";
}

# --- Helper Functions ---
function Test-PythonVersion {
    try {
        $versionString = (python --version 2>&1).Split(' ')[1]
        $version = [System.Version]$versionString
        if ($version -ge [System.Version]$RequiredPythonVersion) {
            Write-Host "Python version $version found. (OK)" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Python version $version is installed, but version $RequiredPythonVersion or newer is required." -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Python is not installed or not found in PATH. Please install Python $RequiredPythonVersion or newer." -ForegroundColor Red
        return $false
    }
}

function Get-NvidiaGpu {
    Write-Host "Checking for NVIDIA GPU..."
    try {
        $gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.AdapterCompatibility -like '*NVIDIA*' }
        if ($gpu) {
            Write-Host "NVIDIA GPU found: $($gpu.Caption)" -ForegroundColor Green
            return $true
        } else {
            Write-Host "No NVIDIA GPU found." -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "Could not check for GPU. Assuming no NVIDIA GPU is present." -ForegroundColor Yellow
        return $false
    }
}

function Get-CudaVersion {
    Write-Host "Checking for CUDA version..."
    try {
        $nvccOutput = nvcc --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $match = $nvccOutput | Select-String -Pattern "release (\d+\.\d+)"
            if ($match) {
                $version = $match.Matches.Groups[1].Value
                Write-Host "CUDA Toolkit version $version found via nvcc." -ForegroundColor Green
                return $version
            }
        }
        Write-Host "nvcc not found or failed. Make sure the NVIDIA CUDA Toolkit is installed and 'nvcc' is in your PATH." -ForegroundColor Yellow
        return $null
    } catch {
        Write-Host "nvcc command failed. CUDA Toolkit may not be installed or configured correctly." -ForegroundColor Yellow
        return $null
    }
}


# --- Main Script ---
Write-Host "--- ChipTrainer Environment Setup ---"

# 1. Verify Python
Write-Host "Step 1: Checking Python version..."
if (-not (Test-PythonVersion)) {
    Write-Host "Setup cannot continue. Please install the required Python version." -ForegroundColor Red
    exit 1
}

# 2. Create Virtual Environment
if (Test-Path -Path $VenvDir) {
    Write-Host "Step 2: Virtual environment '$VenvDir' already exists. Skipping creation." -ForegroundColor Yellow
} else {
    Write-Host "Step 2: Creating Python virtual environment in '$VenvDir'..."
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
}

# 3. Determine PyTorch version to install (CPU vs GPU)
$PyTorchInstallArgs = @("torch", "torchvision") # Default to CPU
if (Get-NvidiaGpu) {
    $cudaVersion = Get-CudaVersion
    if ($cudaVersion -and $SupportedCudaVersions.ContainsKey($cudaVersion)) {
        $cudaWheel = $SupportedCudaVersions[$cudaVersion]
        $PyTorchIndexUrl = "https://download.pytorch.org/whl/$cudaWheel"
        $PyTorchInstallArgs += "--index-url", $PyTorchIndexUrl
        Write-Host "Setup will install PyTorch with CUDA $cudaVersion support." -ForegroundColor Cyan
    } else {
        Write-Host "Supported CUDA version not found. Falling back to CPU-only PyTorch." -ForegroundColor Yellow
    }
} else {
    Write-Host "Proceeding with CPU-only PyTorch installation." -ForegroundColor Cyan
}

# 4. Install Dependencies
Write-Host "Step 4: Installing dependencies..."
try {
    # Define path to Python executable in the virtual environment
    $PythonExe = Join-Path -Path $PSScriptRoot -ChildPath "$VenvDir\Scripts\python.exe"

    if (-not (Test-Path -Path $PythonExe)) {
        Write-Host "Python executable not found in virtual environment: $PythonExe" -ForegroundColor Red
        Write-Host "Please ensure the virtual environment was created correctly."
        exit 1
    }

    Write-Host "Upgrading pip..."
    & $PythonExe -m pip install --upgrade pip

    Write-Host "Installing PyTorch..."
    & $PythonExe -m pip install $PyTorchInstallArgs

    Write-Host "Installing packages from requirements.txt..."
    if (Test-Path -Path "requirements.txt") {
        & $PythonExe -m pip install -r requirements.txt
    } else {
        Write-Host "requirements.txt not found, skipping." -ForegroundColor Yellow
    }

    Write-Host "All dependencies installed successfully." -ForegroundColor Green
} catch {
    Write-Host "An error occurred during dependency installation." -ForegroundColor Red
    Write-Host $_
    exit 1
}

# 5. Verify Installation
Write-Host "Step 5: Verifying PyTorch installation..."
& $PythonExe "$PSScriptRoot\verify_install.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch installation verification failed. Please check the errors above." -ForegroundColor Red
} else {
    Write-Host "PyTorch installation verified successfully." -ForegroundColor Green
}


Write-Host "--- Setup Complete ---"
Write-Host "To activate the environment in your terminal, run:"
Write-Host ".\.venv\Scripts\Activate.ps1"
