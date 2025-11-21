#!/usr/bin/env python3
"""
train_from_config.py - CLI script to train PyTorch models from JSON configuration

This script accepts a JSON configuration file containing all training parameters
and runs the training process without requiring the GUI.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Import the training thread from the GUI application
try:
    from chip_trainer_gui import TrainingThread, TORCH_AVAILABLE, TORCH_ERROR
except ImportError as e:
    print(f"Error: Failed to import chip_trainer_gui module: {e}")
    sys.exit(1)


def load_config(config_path):
    """Load and validate JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def validate_config(config):
    """Validate that required fields are present in the configuration."""
    required_fields = ['data_path', 'model_name', 'epochs', 'learning_rate', 'batch_size']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        print(f"Error: Missing required fields in configuration: {', '.join(missing_fields)}")
        return False
    
    # Validate data path exists
    if not os.path.exists(config['data_path']):
        print(f"Error: Data path does not exist: {config['data_path']}")
        return False
    
    return True


def get_default_config():
    """Return a default configuration template."""
    return {
        # Required fields
        "data_path": "/path/to/your/dataset",
        "model_name": "ResNet18",
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 32,
        
        # Data configuration
        "train_split": 0.7,
        "val_split": 0.2,
        "test_split": 0.1,
        "data_subset": "all",  # or an integer
        
        # Model loading
        "pretrained_model_path": "",  # Path to checkpoint to continue training
        
        # Advanced options
        "advanced_options": {
            "solver": "adam",  # adam, adamw, sgd, rmsprop, adagrad
            "momentum": 0.9,
            "weight_decay": 0.0,
            "loss": "CrossEntropyLoss",  # CrossEntropyLoss, NLLLoss, BCELoss
            "scheduler": "none",  # none, steplr, reducelronplateau, cosineannealinglr
            "lr_drop_factor": 0.1,
            "lr_drop_period": 10,
            "patience": 0,
            "shuffle": "every-epoch",  # every-epoch, once
            "metric": "Accuracy",  # Accuracy, Precision, Recall, F1-Score, AUROC, etc.
            "fscore_beta": 1.0,
            "hflip": False,
            "vflip": False,
            "rotation": False,
            "checkpoint_enabled": False,
            "checkpoint_threshold": 0.5,
            "checkpoint_dir": "",
            "tracking_enabled": False,
            "tracking_frequency": 1,
            "tracked_metrics": [],  # e.g., ["Accuracy", "Precision", "F1-Score"]
            "tracking_fscore_beta": 1.0
        },
        
        # Output configuration
        "output_dir": "./output",  # Directory to save results
        "export_model": True,  # Whether to export the final model
        "export_metrics": True  # Whether to export metric history
    }


class ConfigLogger:
    """Simple logger that prints to stdout and optionally to a file."""
    def __init__(self, log_file=None):
        self.log_file = log_file
        if self.log_file:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def log(self, message):
        """Print message and write to log file if configured."""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train PyTorch image classification models from JSON configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python train_from_config.py config.json
  python train_from_config.py config.json --output ./results
  python train_from_config.py --generate-template > config_template.json

Configuration file format:
  The JSON file should contain all training parameters including data paths,
  model architecture, hyperparameters, and advanced options. Use --generate-template
  to see a complete example configuration.
        """
    )
    
    parser.add_argument('config', nargs='?', help='Path to JSON configuration file')
    parser.add_argument('--generate-template', action='store_true',
                        help='Generate a template configuration file and print to stdout')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory from config file')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to save training log (default: output_dir/training.log)')
    
    args = parser.parse_args()
    
    # Generate template if requested
    if args.generate_template:
        print(json.dumps(get_default_config(), indent=2))
        return 0
    
    # Check if config file provided
    if not args.config:
        parser.print_help()
        print("\nError: Configuration file required (or use --generate-template)")
        return 1
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        print(f"Error: PyTorch is not available: {TORCH_ERROR}")
        return 1
    
    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    if not validate_config(config):
        return 1
    
    print("Configuration loaded successfully.")
    
    # Extract configuration values
    data_path = config['data_path']
    model_name = config['model_name']
    epochs = config['epochs']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    
    splits = (
        config.get('train_split', 0.7),
        config.get('val_split', 0.2),
        config.get('test_split', 0.1)
    )
    
    data_subset = config.get('data_subset', 'all')
    pretrained_model_path = config.get('pretrained_model_path', '')
    advanced_options = config.get('advanced_options', {})
    
    # Setup output directory
    output_dir = args.output if args.output else config.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = args.log_file if args.log_file else os.path.join(output_dir, 'training.log')
    logger = ConfigLogger(log_file)
    
    logger.log("=" * 60)
    logger.log(f"ChipTrainer CLI - Training Started")
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)
    logger.log(f"Data Path: {data_path}")
    logger.log(f"Model: {model_name}")
    logger.log(f"Epochs: {epochs}")
    logger.log(f"Learning Rate: {lr}")
    logger.log(f"Batch Size: {batch_size}")
    logger.log(f"Splits: Train={splits[0]}, Val={splits[1]}, Test={splits[2]}")
    logger.log(f"Output Directory: {output_dir}")
    logger.log(f"Log File: {log_file}")
    logger.log("=" * 60)
    
    # Create training thread
    training_thread = TrainingThread(
        data_path=data_path,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        splits=splits,
        data_subset=data_subset,
        advanced_options=advanced_options,
        pretrained_model_path=pretrained_model_path
    )
    
    # Connect logging signal
    training_thread.log.connect(logger.log)
    
    # Variable to store results
    training_result = {'model': None, 'metric_info': None}
    
    def on_training_finished(model, metric_info):
        training_result['model'] = model
        training_result['metric_info'] = metric_info
    
    training_thread.finished.connect(on_training_finished)
    
    # Start training (run synchronously in main thread for CLI)
    training_thread.run()
    
    # Process results
    model = training_result['model']
    metric_info = training_result['metric_info']
    
    if model is None:
        logger.log("\n" + "=" * 60)
        logger.log("Training failed or was stopped.")
        logger.log("=" * 60)
        return 1
    
    logger.log("\n" + "=" * 60)
    logger.log("Training completed successfully!")
    logger.log("=" * 60)
    
    # Export model if requested
    if config.get('export_model', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_arch = metric_info.get('model_name', model_name)
        metric_name = metric_info.get('metric_name', 'accuracy')
        metric_value = metric_info.get('metric_value', 0.0)
        
        # Format metric name for filename
        metric_name_clean = metric_name.lower().replace('-', '').replace(' ', '')
        
        # Check if F-Score with custom beta
        beta = metric_info.get('beta', 1.0)
        if metric_name == "F1-Score" and beta != 1.0:
            metric_name_clean = f"f{beta:.1f}score".replace('.', '_')
        
        model_filename = f"{model_arch}_final_{metric_name_clean}{metric_value:.2f}_{timestamp}.pth"
        model_path = os.path.join(output_dir, model_filename)
        
        try:
            import torch
            torch.save(model.state_dict(), model_path)
            logger.log(f"Model exported to: {model_path}")
        except Exception as e:
            logger.log(f"Error exporting model: {e}")
    
    # Export metrics if requested
    if config.get('export_metrics', True) and metric_info.get('metric_history'):
        metric_history = metric_info['metric_history']
        
        if metric_history.get('epochs'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_filename = f"{model_arch}_metrics_{timestamp}.json"
            metrics_path = os.path.join(output_dir, metrics_filename)
            
            # Prepare export data
            export_data = metric_history.copy()
            if metric_info.get('confusion_matrix') is not None:
                export_data['confusion_matrix'] = metric_info['confusion_matrix']
                export_data['class_names'] = metric_info.get('class_names', [])
            
            try:
                with open(metrics_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                logger.log(f"Metrics exported to: {metrics_path}")
            except Exception as e:
                logger.log(f"Error exporting metrics: {e}")
    
    logger.log("=" * 60)
    logger.log("All operations completed.")
    logger.log("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
