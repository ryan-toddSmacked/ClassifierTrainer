#!/usr/bin/env python3
"""
hyperparameter_search.py - Automated hyperparameter optimization for ChipTrainer

This script performs systematic hyperparameter search to find the best model configuration.
It trains multiple models with different hyperparameter combinations and tracks the best performing ones.
"""

import argparse
import json
import sys
import os
import itertools
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Try to import optuna for smart search
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import the training thread from the GUI application
try:
    from chip_trainer_gui import TrainingThread, TORCH_AVAILABLE, TORCH_ERROR
except ImportError as e:
    print(f"Error: Failed to import chip_trainer_gui module: {e}")
    sys.exit(1)


class HyperparameterSearchConfig:
    """Configuration for hyperparameter search space."""
    
    # Define all possible hyperparameter search spaces
    SEARCH_SPACES = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64, 128],
        'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
        'weight_decay': [0.0, 0.0001, 0.001, 0.01],
        'momentum': [0.9, 0.95, 0.99],  # For SGD
        'scheduler': ['none', 'steplr', 'reducelronplateau', 'cosineannealinglr'],
        'lr_drop_factor': [0.1, 0.5, 0.75],
        'lr_drop_period': [5, 10, 15, 20],
        'hflip': [False, True],
        'vflip': [False, True],
        'rotation': [False, True],
    }
    
    @staticmethod
    def get_default_enabled():
        """Return default enabled hyperparameters for search."""
        return ['learning_rate', 'batch_size', 'optimizer']
    
    @staticmethod
    def suggest_hyperparameter(trial, param_name):
        """Suggest a hyperparameter value for Optuna trial."""
        if param_name == 'learning_rate':
            return trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        elif param_name == 'batch_size':
            return trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        elif param_name == 'optimizer':
            return trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop'])
        elif param_name == 'weight_decay':
            return trial.suggest_float('weight_decay', 0.0, 0.01, log=True)
        elif param_name == 'momentum':
            return trial.suggest_float('momentum', 0.9, 0.99)
        elif param_name == 'scheduler':
            return trial.suggest_categorical('scheduler', ['none', 'steplr', 'reducelronplateau', 'cosineannealinglr'])
        elif param_name == 'lr_drop_factor':
            return trial.suggest_categorical('lr_drop_factor', [0.1, 0.5, 0.75])
        elif param_name == 'lr_drop_period':
            return trial.suggest_categorical('lr_drop_period', [5, 10, 15, 20])
        elif param_name == 'hflip':
            return trial.suggest_categorical('hflip', [False, True])
        elif param_name == 'vflip':
            return trial.suggest_categorical('vflip', [False, True])
        elif param_name == 'rotation':
            return trial.suggest_categorical('rotation', [False, True])
        else:
            return None


def generate_metric_plots(metric_history, model_name, output_dir, trial_num, metric_value, timestamp):
    """Generate matplotlib figures for tracked metrics.
    
    Args:
        metric_history: Dictionary containing epochs, train, and val metrics
        model_name: Name of the model architecture
        output_dir: Directory to save plots
        trial_num: Trial number for filename
        metric_value: Best metric value achieved
        timestamp: Timestamp for filename
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available for generating plots.")
        return []
    
    try:
        epochs = metric_history.get("epochs", [])
        train_metrics = metric_history.get("train", {})
        val_metrics = metric_history.get("val", {})
        
        if not epochs or not train_metrics:
            return []
        
        saved_plots = []
        
        # Create a plot for each metric
        for metric_name in train_metrics.keys():
            if metric_name not in val_metrics:
                continue
            
            # Format display name for F-Score with correct beta
            display_name = metric_name
            if metric_name == "F1-Score":
                tracking_beta = metric_history.get("tracking_beta", 1.0)
                if tracking_beta == 1.0:
                    display_name = "F1-Score"
                else:
                    display_name = f"F{tracking_beta:.1f}-Score"
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_metrics[metric_name], 'b-o', label='Train', linewidth=2, markersize=6)
            plt.plot(epochs, val_metrics[metric_name], 'r-s', label='Validation', linewidth=2, markersize=6)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(f'{display_name} (%)', fontsize=12)
            plt.title(f'{model_name} - Trial {trial_num} - {display_name} over Training', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plot_filename = f"trial{trial_num}_{metric_name.replace('-', '').replace(' ', '_')}_{metric_value:.2f}_{timestamp}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_plots.append(plot_path)
        
        return saved_plots
        
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")
        return []


def load_base_config(config_path):
    """Load base configuration from JSON file."""
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


def generate_search_configs(base_config, enabled_params, search_method='grid', max_trials=None):
    """
    Generate configurations for hyperparameter search.
    
    Args:
        base_config: Base configuration dictionary
        enabled_params: List of hyperparameter names to search
        search_method: 'grid' or 'random'
        max_trials: Maximum number of trials (for random search)
    
    Returns:
        List of configuration dictionaries
    """
    search_space = {}
    for param in enabled_params:
        if param in HyperparameterSearchConfig.SEARCH_SPACES:
            search_space[param] = HyperparameterSearchConfig.SEARCH_SPACES[param]
    
    if not search_space:
        print("Warning: No hyperparameters enabled for search. Using base configuration only.")
        return [base_config]
    
    configs = []
    
    if search_method == 'grid':
        # Grid search: all combinations
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        
        for combination in itertools.product(*values):
            config = base_config.copy()
            config['advanced_options'] = config.get('advanced_options', {}).copy()
            
            for key, value in zip(keys, combination):
                if key in ['learning_rate', 'batch_size']:
                    config[key] = value
                else:
                    config['advanced_options'][key] = value
            
            configs.append(config)
    
    elif search_method == 'random':
        # Random search: sample random combinations
        if max_trials is None:
            max_trials = 20
        
        for _ in range(max_trials):
            config = base_config.copy()
            config['advanced_options'] = config.get('advanced_options', {}).copy()
            
            for param in enabled_params:
                if param in search_space:
                    value = random.choice(search_space[param])
                    if param in ['learning_rate', 'batch_size']:
                        config[param] = value
                    else:
                        config['advanced_options'][param] = value
            
            configs.append(config)
    
    return configs


def config_to_string(config):
    """Convert config to a readable string for logging."""
    parts = []
    parts.append(f"lr={config.get('learning_rate', 0.001)}")
    parts.append(f"bs={config.get('batch_size', 32)}")
    
    adv = config.get('advanced_options', {})
    parts.append(f"opt={adv.get('optimizer', 'adam')}")
    
    if adv.get('weight_decay', 0.0) > 0:
        parts.append(f"wd={adv.get('weight_decay')}")
    
    if adv.get('scheduler', 'none') != 'none':
        parts.append(f"sched={adv.get('scheduler')}")
    
    aug = []
    if adv.get('hflip'): aug.append('hflip')
    if adv.get('vflip'): aug.append('vflip')
    if adv.get('rotation'): aug.append('rot')
    if aug:
        parts.append(f"aug=[{','.join(aug)}]")
    
    return ", ".join(parts)


class SearchLogger:
    """Logger for hyperparameter search progress."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write(f"Hyperparameter Search Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message):
        """Print and write to log file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
    
    def log_trial(self, trial_num, total_trials, config_str, metric_value, metric_name):
        """Log trial results."""
        message = f"Trial {trial_num}/{total_trials}: {config_str} -> {metric_name}={metric_value:.2f}%"
        self.log(message)


def run_search(args):
    """Execute hyperparameter search."""
    
    # Load base configuration
    print(f"Loading base configuration from: {args.config}")
    base_config = load_base_config(args.config)
    
    # Validate base config
    if not os.path.exists(base_config['data_path']):
        print(f"Error: Data path does not exist: {base_config['data_path']}")
        return 1
    
    # Setup output directory
    output_dir = args.output if args.output else './hyperparameter_search_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'search_log_{timestamp}.txt')
    logger = SearchLogger(log_file)
    
    logger.log("=" * 80)
    logger.log("ChipTrainer - Hyperparameter Search")
    logger.log("=" * 80)
    logger.log(f"Base Config: {args.config}")
    logger.log(f"Data Path: {base_config['data_path']}")
    logger.log(f"Model: {base_config['model_name']}")
    logger.log(f"Search Method: {args.method}")
    logger.log(f"Enabled Parameters: {', '.join(args.params)}")
    logger.log(f"Output Directory: {output_dir}")
    logger.log("=" * 80 + "\n")
    
    # Generate search configurations
    if args.method == 'random':
        configs = generate_search_configs(base_config, args.params, 'random', args.max_trials)
    else:
        configs = generate_search_configs(base_config, args.params, 'grid')
    
    total_trials = len(configs)
    logger.log(f"Generated {total_trials} configurations to test\n")
    
    # Track results
    results = []
    best_metric = -float('inf')
    best_config = None
    best_model_path = None
    
    # Run trials
    for trial_num, config in enumerate(configs, 1):
        logger.log(f"\n{'=' * 80}")
        logger.log(f"Starting Trial {trial_num}/{total_trials}")
        logger.log(f"Configuration: {config_to_string(config)}")
        logger.log('=' * 80)
        
        # Extract training parameters
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
        
        # Connect logging
        training_thread.log.connect(lambda msg: print(f"  {msg}"))
        
        # Store result
        training_result = {'model': None, 'metric_info': None}
        
        def on_training_finished(model, metric_info):
            training_result['model'] = model
            training_result['metric_info'] = metric_info
        
        training_thread.finished.connect(on_training_finished)
        
        # Run training
        training_thread.run()
        
        # Process results
        model = training_result['model']
        metric_info = training_result['metric_info']
        
        if model is None:
            logger.log(f"Trial {trial_num} FAILED")
            results.append({
                'trial': trial_num,
                'config': config_to_string(config),
                'status': 'failed',
                'metric_value': 0.0
            })
            continue
        
        # Extract metric value
        metric_name = metric_info.get('metric_name', 'Accuracy')
        metric_value = metric_info.get('metric_value', 0.0)
        
        logger.log_trial(trial_num, total_trials, config_to_string(config), metric_value, metric_name)
        
        # Save model if it's the best so far
        if metric_value > best_metric:
            best_metric = metric_value
            best_config = config
            
            # Save best model
            model_filename = f"best_model_trial{trial_num}_{metric_name.lower().replace('-', '')}_{metric_value:.2f}_{timestamp}.pth"
            model_path = os.path.join(output_dir, model_filename)
            
            try:
                import torch
                torch.save(model.state_dict(), model_path)
                best_model_path = model_path
                logger.log(f"  *** NEW BEST MODEL *** Saved to: {model_filename}")
            except Exception as e:
                logger.log(f"  Error saving model: {e}")
        
        # Store trial results
        trial_result = {
            'trial': trial_num,
            'config': config,
            'config_str': config_to_string(config),
            'metric_name': metric_name,
            'metric_value': metric_value,
            'metric_history': metric_info.get('metric_history', {}),
            'status': 'success'
        }
        results.append(trial_result)
        
        # Save incremental results
        results_file = os.path.join(output_dir, f'search_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump({
                'search_method': args.method,
                'enabled_params': args.params,
                'trials': results,
                'best_trial': {
                    'metric_value': best_metric,
                    'config': best_config,
                    'config_str': config_to_string(best_config) if best_config else None,
                    'model_path': best_model_path
                }
            }, f, indent=2)
    
    # Final summary
    logger.log("\n" + "=" * 80)
    logger.log("SEARCH COMPLETE")
    logger.log("=" * 80)
    logger.log(f"Total trials: {total_trials}")
    logger.log(f"Successful trials: {sum(1 for r in results if r.get('status') == 'success')}")
    logger.log(f"Failed trials: {sum(1 for r in results if r.get('status') == 'failed')}")
    logger.log(f"\nBest {metric_name}: {best_metric:.2f}%")
    logger.log(f"Best configuration: {config_to_string(best_config)}")
    logger.log(f"Best model saved to: {best_model_path}")
    logger.log(f"\nFull results saved to: {results_file}")
    logger.log("=" * 80)
    
    # Save best config for easy reuse
    if best_config:
        best_config_file = os.path.join(output_dir, f'best_config_{timestamp}.json')
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        logger.log(f"Best configuration saved to: {best_config_file}")
    
    # Generate plots for top 3 models if tracking is enabled
    if base_config.get('advanced_options', {}).get('tracking_enabled', False):
        logger.log("\nGenerating metric tracking plots for top 3 models...")
        
        # Sort successful results by metric value
        successful_results = [r for r in results if r.get('status') == 'success' and r.get('metric_history')]
        successful_results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Generate plots for top 3
        top_n = min(3, len(successful_results))
        for i, result in enumerate(successful_results[:top_n], 1):
            trial_num = result['trial']
            metric_value = result['metric_value']
            metric_history = result['metric_history']
            model_name = base_config['model_name']
            
            logger.log(f"  Generating plots for Trial {trial_num} (Rank #{i}, {metric_name}: {metric_value:.2f}%)...")
            plots = generate_metric_plots(metric_history, model_name, output_dir, trial_num, metric_value, timestamp)
            if plots:
                logger.log(f"    Saved {len(plots)} metric plots")
        
        logger.log(f"Metric tracking plots generated for top {top_n} models")
    
    return 0


def run_smart_search(args):
    """Execute smart hyperparameter search using Optuna."""
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is not installed. Install it with: pip install optuna")
        print("Falling back to random search...")
        args.method = 'random'
        return run_search(args)
    
    # Load base configuration
    print(f"Loading base configuration from: {args.config}")
    base_config = load_base_config(args.config)
    
    # Validate base config
    if not os.path.exists(base_config['data_path']):
        print(f"Error: Data path does not exist: {base_config['data_path']}")
        return 1
    
    # Setup output directory
    output_dir = args.output if args.output else './hyperparameter_search_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'search_log_smart_{timestamp}.txt')
    logger = SearchLogger(log_file)
    
    logger.log("=" * 80)
    logger.log("ChipTrainer - Smart Hyperparameter Search (Optuna)")
    logger.log("=" * 80)
    logger.log(f"Base Config: {args.config}")
    logger.log(f"Data Path: {base_config['data_path']}")
    logger.log(f"Model: {base_config['model_name']}")
    logger.log(f"Max Trials: {args.max_trials}")
    logger.log(f"Enabled Parameters: {', '.join(args.params)}")
    logger.log(f"Output Directory: {output_dir}")
    logger.log("=" * 80 + "\n")
    
    # Track best model globally and all trial results
    best_model_info = {'path': None, 'metric': -float('inf'), 'config': None}
    all_trial_results = []  # Store all trial results with metric history
    trial_counter = [0]  # Use list to make it mutable in nested function
    
    def objective(trial):
        """Optuna objective function."""
        trial_counter[0] += 1
        trial_num = trial_counter[0]
        
        # Create config for this trial
        config = base_config.copy()
        config['advanced_options'] = config.get('advanced_options', {}).copy()
        
        # Suggest hyperparameters
        for param in args.params:
            value = HyperparameterSearchConfig.suggest_hyperparameter(trial, param)
            if value is not None:
                if param in ['learning_rate', 'batch_size']:
                    config[param] = value
                else:
                    config['advanced_options'][param] = value
        
        logger.log(f"\n{'=' * 80}")
        logger.log(f"Trial {trial_num}/{args.max_trials}")
        logger.log(f"Configuration: {config_to_string(config)}")
        logger.log('=' * 80)
        
        # Extract training parameters
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
        
        # Connect logging
        training_thread.log.connect(lambda msg: print(f"  {msg}"))
        
        # Store result
        training_result = {'model': None, 'metric_info': None}
        
        def on_training_finished(model, metric_info):
            training_result['model'] = model
            training_result['metric_info'] = metric_info
        
        training_thread.finished.connect(on_training_finished)
        
        # Run training
        training_thread.run()
        
        # Process results
        model = training_result['model']
        metric_info = training_result['metric_info']
        
        if model is None:
            logger.log(f"Trial {trial_num} FAILED")
            return 0.0  # Return poor score for failed trials
        
        # Extract metric value
        metric_name = metric_info.get('metric_name', 'Accuracy')
        metric_value = metric_info.get('metric_value', 0.0)
        metric_history = metric_info.get('metric_history', {})
        
        # Store trial result with metric history and model
        all_trial_results.append({
            'trial_number': trial_num,
            'metric_value': metric_value,
            'metric_history': metric_history,
            'metric_name': metric_name,
            'model': model,
            'config': config,
            'params': trial.params.copy()
        })
        
        # Track best for final config export
        if metric_value > best_model_info['metric']:
            best_model_info['metric'] = metric_value
            best_model_info['config'] = config
        
        logger.log_trial(trial_num, args.max_trials, config_to_string(config), metric_value, metric_name)
        
        return metric_value
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name=f'chiptrainer_{timestamp}',
        sampler=optuna.samplers.TPESampler(seed=42)  # TPE (Tree-structured Parzen Estimator)
    )
    
    # Run optimization
    logger.log(f"Starting Optuna optimization with up to {args.max_trials} trials\n")
    study.optimize(objective, n_trials=args.max_trials, show_progress_bar=False)
    
    # Save top 3 models
    logger.log("\n" + "=" * 80)
    logger.log("Saving top 3 models...")
    logger.log("=" * 80)
    
    # Sort trial results by metric value
    all_trial_results.sort(key=lambda x: x['metric_value'], reverse=True)
    
    # Save top 3 models
    import torch
    top_n = min(3, len(all_trial_results))
    saved_model_paths = []
    
    for rank, result in enumerate(all_trial_results[:top_n], 1):
        trial_num = result['trial_number']
        metric_value = result['metric_value']
        metric_name = result['metric_name']
        model = result['model']
        
        model_filename = f"best_model_smart_trial{trial_num}_rank{rank}_{metric_name.lower().replace('-', '')}_{metric_value:.2f}_{timestamp}.pth"
        model_path = os.path.join(output_dir, model_filename)
        
        try:
            torch.save(model.state_dict(), model_path)
            saved_model_paths.append(model_path)
            logger.log(f"Rank #{rank}: Trial {trial_num} ({metric_name}: {metric_value:.2f}%) -> {model_filename}")
        except Exception as e:
            logger.log(f"Error saving model for trial {trial_num}: {e}")
    
    # Update best_model_info with the #1 ranked model
    if saved_model_paths:
        best_model_info['path'] = saved_model_paths[0]
    
    logger.log("=" * 80 + "\n")
    
    # Evaluate top 3 models on test set and generate confusion matrices
    logger.log("=" * 80)
    logger.log("Evaluating top 3 models on test set...")
    logger.log("=" * 80)
    
    # Prepare test dataset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    try:
        data_path = base_config['data_path']
        batch_size = base_config.get('batch_size', 32)
        
        # Load full dataset to get splits
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = datasets.ImageFolder(data_path, transform=transform)
        class_names = full_dataset.classes
        num_classes = len(class_names)
        
        # Calculate split sizes
        train_split = base_config.get('train_split', 0.7)
        val_split = base_config.get('val_split', 0.2)
        test_split = base_config.get('test_split', 0.1)
        
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        from torch.utils.data import random_split
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.log(f"Test set size: {test_size} images")
        logger.log(f"Number of classes: {num_classes}")
        logger.log("")
        
        # Evaluate each top model
        for rank, result in enumerate(all_trial_results[:top_n], 1):
            trial_num = result['trial_number']
            metric_value = result['metric_value']
            metric_name = result['metric_name']
            model = result['model']
            
            logger.log(f"Evaluating Rank #{rank} (Trial {trial_num})...")
            
            # Set model to eval mode
            model.eval()
            device = next(model.parameters()).device
            
            # Collect predictions
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Create confusion matrix
            import numpy as np
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
            for true_label, pred_label in zip(all_labels, all_preds):
                confusion_matrix[true_label][pred_label] += 1
            
            # Generate confusion matrix plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(max(8, num_classes), max(8, num_classes)))
                im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                # Set ticks and labels
                ax.set(xticks=np.arange(num_classes),
                       yticks=np.arange(num_classes),
                       xticklabels=class_names,
                       yticklabels=class_names,
                       xlabel='Predicted Label',
                       ylabel='True Label',
                       title=f'Trial {trial_num} (Rank #{rank}) - Confusion Matrix (Test Set)')
                
                # Rotate the tick labels and set their alignment
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Loop over data dimensions and create text annotations
                thresh = confusion_matrix.max() / 2.
                for i in range(num_classes):
                    for j in range(num_classes):
                        ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if confusion_matrix[i, j] > thresh else "black")
                
                fig.tight_layout()
                
                # Save figure
                cm_filename = f"trial{trial_num}_rank{rank}_confusion_matrix_{timestamp}.png"
                cm_path = os.path.join(output_dir, cm_filename)
                plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.log(f"  Confusion matrix saved: {cm_filename}")
                
            except Exception as e:
                logger.log(f"  Error generating confusion matrix plot: {e}")
        
        logger.log("=" * 80 + "\n")
        
    except Exception as e:
        logger.log(f"Error evaluating models on test set: {e}")
        logger.log("=" * 80 + "\n")
    
    # Get results
    best_trial = study.best_trial
    
    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        })
    
    results_file = os.path.join(output_dir, f'search_results_smart_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'search_method': 'smart (Optuna TPE)',
            'enabled_params': args.params,
            'total_trials': len(study.trials),
            'trials': trials_data,
            'best_trial': {
                'number': best_trial.number,
                'metric_value': best_trial.value,
                'params': best_trial.params,
                'config': best_model_info['config'],
                'model_path': best_model_info['path']
            }
        }, f, indent=2)
    
    # Final summary
    logger.log("\n" + "=" * 80)
    logger.log("SMART SEARCH COMPLETE")
    logger.log("=" * 80)
    logger.log(f"Total trials: {len(study.trials)}")
    logger.log(f"Best metric value: {best_trial.value:.2f}%")
    logger.log(f"Best parameters: {best_trial.params}")
    logger.log(f"Best model saved to: {best_model_info['path']}")
    logger.log(f"\nFull results saved to: {results_file}")
    logger.log("=" * 80)
    
    # Save best config
    if best_model_info['config']:
        best_config_file = os.path.join(output_dir, f'best_config_smart_{timestamp}.json')
        with open(best_config_file, 'w') as f:
            json.dump(best_model_info['config'], f, indent=2)
        logger.log(f"Best configuration saved to: {best_config_file}")
    
    # Generate optimization history plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Get metric name from config
        metric_name = base_config.get('advanced_options', {}).get('metric', 'Accuracy')
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        
        # Determine if fig is a Figure or Axes object
        if hasattr(fig, 'set_ylabel'):
            # fig is an Axes object, get the figure from it
            ax = fig
            fig = ax.get_figure()
        else:
            # fig is a Figure object, get the axes
            try:
                if isinstance(fig.axes, list) and len(fig.axes) > 0:
                    ax = fig.axes[0]
                else:
                    ax = fig.axes
            except (AttributeError, TypeError):
                ax = plt.gca()
        
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'Optimization History - {metric_name}')
        
        plot_file = os.path.join(output_dir, f'optimization_history_{timestamp}.png')
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.log(f"Optimization history plot saved to: {plot_file}")
    except Exception as e:
        logger.log(f"Could not generate optimization history plot: {e}")
    
    # Generate plots for top 3 models if tracking is enabled
    if base_config.get('advanced_options', {}).get('tracking_enabled', False):
        logger.log("\nGenerating metric tracking plots for top 3 models...")
        
        # Sort trial results by metric value
        all_trial_results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Generate plots for top 3
        top_n = min(3, len(all_trial_results))
        metric_name = base_config.get('advanced_options', {}).get('metric', 'Accuracy')
        
        for i, result in enumerate(all_trial_results[:top_n], 1):
            trial_num = result['trial_number']  # Already 1-indexed
            metric_value = result['metric_value']
            metric_history = result['metric_history']
            model_name = base_config['model_name']
            
            if not metric_history.get('epochs'):
                continue  # Skip if no tracking data
            
            logger.log(f"  Generating plots for Trial {trial_num} (Rank #{i}, {metric_name}: {metric_value:.2f}%)...")
            plots = generate_metric_plots(metric_history, model_name, output_dir, trial_num, metric_value, timestamp)
            if plots:
                logger.log(f"    Saved {len(plots)} metric plots")
        
        logger.log(f"Metric tracking plots generated for top {top_n} models")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Automated hyperparameter search for ChipTrainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Grid search with learning rate and batch size
  python hyperparameter_search.py config.json --params learning_rate batch_size

  # Random search with 30 trials
  python hyperparameter_search.py config.json --method random --max-trials 30 \
      --params learning_rate batch_size optimizer scheduler

  # Smart search (Bayesian optimization with Optuna)
  python hyperparameter_search.py config.json --method smart --max-trials 50 \
      --params learning_rate batch_size optimizer weight_decay scheduler

  # Search all augmentation options
  python hyperparameter_search.py config.json --params hflip vflip rotation

Available hyperparameters:
  learning_rate, batch_size, optimizer, weight_decay, momentum, scheduler,
  lr_drop_factor, lr_drop_period, hflip, vflip, rotation

Search methods:
  grid   - Try all combinations (exhaustive but can be slow)
  random - Try random combinations (faster, good for large spaces)
  smart  - Bayesian optimization with Optuna (learns from trials, most efficient)
           Requires: pip install optuna
        """
    )
    
    parser.add_argument('config', help='Path to base JSON configuration file')
    parser.add_argument('--params', nargs='+', required=True,
                        choices=list(HyperparameterSearchConfig.SEARCH_SPACES.keys()),
                        help='Hyperparameters to search')
    parser.add_argument('--method', choices=['grid', 'random', 'smart'], default='grid',
                        help='Search method: grid (exhaustive), random (sampling), smart (Bayesian/Optuna)')
    parser.add_argument('--max-trials', type=int, default=20,
                        help='Maximum trials for random/smart search (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results (default: ./hyperparameter_search_results)')
    
    args = parser.parse_args()
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        print(f"Error: PyTorch is not available: {TORCH_ERROR}")
        return 1
    
    # Run appropriate search method
    if args.method == 'smart':
        return run_smart_search(args)
    else:
        return run_search(args)


if __name__ == "__main__":
    sys.exit(main())
