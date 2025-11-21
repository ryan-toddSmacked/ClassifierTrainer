# verify_install.py - Checks the PyTorch installation and CUDA status.

import sys

try:
    import torch
    print(f"Successfully imported PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA is available.")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print("\nPyTorch GPU installation appears to be working correctly.")
    else:
        print("CUDA is not available.")
        print("\nPyTorch CPU installation appears to be working correctly.")

    sys.exit(0)

except ImportError:
    print("Error: Failed to import PyTorch.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)
