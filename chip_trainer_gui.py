"""
ChipTrainer - A Simple PyTorch Image Classifier GUI

This script provides a user-friendly graphical interface for training
popular deep learning models on image classification tasks.
"""
import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime

# Import PyTorch and related libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
    TORCH_ERROR = None
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

# Import torchmetrics
try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

# Import PyQt5 for the GUI
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                 QTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox,
                                 QGroupBox, QGridLayout, QMessageBox, QComboBox,
                                 QDialog, QDialogButtonBox, QCheckBox,
                                 QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
                                 QDialog, QDialogButtonBox)
    from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt5.QtGui import QFont, QTextCursor
except ImportError as e:
    print(f"PyQt5 is required. Please install it: pip install PyQt5. Error: {e}")
    sys.exit(1)

# Import GPUtil for GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class AdvancedOptionsDialog(QDialog):
    """A dialog to configure advanced training options."""
    def __init__(self, parent=None, current_options=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Training Options")
        self.setMinimumWidth(400)
        
        if current_options is None:
            current_options = self.get_default_options()
        self.options = current_options

        self.layout = QGridLayout(self)
        
        # --- Optimizer Section ---
        optimizer_group = QGroupBox("Optimizer")
        optimizer_layout = QGridLayout(optimizer_group)
        
        optimizer_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["adam", "adamw", "sgdm", "rmsprop", "adagrad"])
        self.solver_combo.setCurrentText(self.options.get("solver", "adam"))
        optimizer_layout.addWidget(self.solver_combo, 0, 1)

        self.momentum_label = QLabel("Momentum:")
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setDecimals(2)
        self.momentum_spin.setValue(self.options.get("momentum", 0.9))
        optimizer_layout.addWidget(self.momentum_label, 1, 0)
        optimizer_layout.addWidget(self.momentum_spin, 1, 1)

        self.weight_decay_label = QLabel("Weight Decay:")
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setValue(self.options.get("weight_decay", 0.0))
        optimizer_layout.addWidget(self.weight_decay_label, 2, 0)
        optimizer_layout.addWidget(self.weight_decay_spin, 2, 1)

        self.layout.addWidget(optimizer_group, 0, 0, 1, 2)

        # --- Loss and Scheduler Section ---
        loss_sched_group = QGroupBox("Loss & Learning Rate Scheduler")
        loss_sched_layout = QGridLayout(loss_sched_group)

        loss_sched_layout.addWidget(QLabel("Loss Function:"), 0, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["CrossEntropyLoss", "NLLLoss", "BCELoss"])
        self.loss_combo.setCurrentText(self.options.get("loss", "CrossEntropyLoss"))
        loss_sched_layout.addWidget(self.loss_combo, 0, 1)

        loss_sched_layout.addWidget(QLabel("LR Scheduler:"), 1, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["none", "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"])
        self.scheduler_combo.setCurrentText(self.options.get("scheduler", "none"))
        loss_sched_layout.addWidget(self.scheduler_combo, 1, 1)

        self.lr_drop_factor_label = QLabel("LR Factor:")
        self.lr_drop_factor_spin = QDoubleSpinBox()
        self.lr_drop_factor_spin.setRange(0.01, 0.99)
        self.lr_drop_factor_spin.setDecimals(2)
        self.lr_drop_factor_spin.setValue(self.options.get("lr_drop_factor", 0.1))
        loss_sched_layout.addWidget(self.lr_drop_factor_label, 2, 0)
        loss_sched_layout.addWidget(self.lr_drop_factor_spin, 2, 1)

        self.lr_drop_period_label = QLabel("LR Period/Patience:")
        self.lr_drop_period_spin = QSpinBox()
        self.lr_drop_period_spin.setRange(1, 100)
        self.lr_drop_period_spin.setValue(self.options.get("lr_drop_period", 10))
        loss_sched_layout.addWidget(self.lr_drop_period_label, 3, 0)
        loss_sched_layout.addWidget(self.lr_drop_period_spin, 3, 1)

        self.layout.addWidget(loss_sched_group, 1, 0, 1, 2)

        # --- Augmentation and Misc Section ---
        aug_misc_group = QGroupBox("Augmentation & Miscellaneous")
        aug_misc_layout = QGridLayout(aug_misc_group)

        self.hflip_check = QCheckBox("Random Horizontal Flip")
        self.hflip_check.setChecked(self.options.get("hflip", False))
        aug_misc_layout.addWidget(self.hflip_check, 0, 0)

        self.vflip_check = QCheckBox("Random Vertical Flip")
        self.vflip_check.setChecked(self.options.get("vflip", False))
        aug_misc_layout.addWidget(self.vflip_check, 0, 1)

        self.rotation_check = QCheckBox("Random Rotation")
        self.rotation_check.setChecked(self.options.get("rotation", False))
        aug_misc_layout.addWidget(self.rotation_check, 1, 0)

        aug_misc_layout.addWidget(QLabel("Class Balance Ratio:"), 2, 0)
        self.class_balance_spin = QDoubleSpinBox()
        self.class_balance_spin.setRange(1.0, 100.0)
        self.class_balance_spin.setDecimals(1)
        self.class_balance_spin.setSingleStep(0.5)
        self.class_balance_spin.setValue(self.options.get("class_balance_ratio", 0.0))
        self.class_balance_spin.setSpecialValueText("Disabled")
        self.class_balance_spin.setToolTip("Maximum ratio between largest and smallest class (e.g., 2.0 = max 2x difference). Set to 0 or 1 to disable.")
        aug_misc_layout.addWidget(self.class_balance_spin, 2, 1)

        aug_misc_layout.addWidget(QLabel("Validation Patience:"), 3, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setToolTip("Epochs to wait for improvement before stopping. 0 to disable.")
        self.patience_spin.setValue(self.options.get("patience", 0))
        aug_misc_layout.addWidget(self.patience_spin, 3, 1)

        aug_misc_layout.addWidget(QLabel("Shuffle Data:"), 4, 0)
        self.shuffle_combo = QComboBox()
        self.shuffle_combo.addItems(["every-epoch", "once"])
        self.shuffle_combo.setCurrentText(self.options.get("shuffle", "every-epoch"))
        aug_misc_layout.addWidget(self.shuffle_combo, 4, 1)

        aug_misc_layout.addWidget(QLabel("Evaluation Metric:"), 5, 0)
        self.metric_combo = QComboBox()
        
        # Populate with available metrics
        if TORCHMETRICS_AVAILABLE:
            metrics_list = [
                "Accuracy", "Precision", "Recall", "F1-Score", "AUROC", 
                "AveragePrecision", "Specificity", "MatthewsCorrCoef", 
                "CohenKappa", "JaccardIndex", "Dice", "HammingDistance"
            ]
        else:
            metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
        
        self.metric_combo.addItems(metrics_list)
        self.metric_combo.setCurrentText(self.options.get("metric", "Accuracy"))
        self.metric_combo.setToolTip("Metric used to evaluate model performance and save checkpoints")
        aug_misc_layout.addWidget(self.metric_combo, 5, 1)

        self.beta_label = QLabel("F-Score Beta:")
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.1, 10.0)
        self.beta_spin.setDecimals(1)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setValue(self.options.get("fscore_beta", 1.0))
        self.beta_spin.setToolTip("Beta parameter for F-Score (beta=1 is F1, beta=2 is F2, etc.)")
        aug_misc_layout.addWidget(self.beta_label, 6, 0)
        aug_misc_layout.addWidget(self.beta_spin, 6, 1)
        
        self.layout.addWidget(aug_misc_group, 2, 0, 1, 2)

        # --- Checkpointing Section ---
        checkpoint_group = QGroupBox("Model Checkpointing")
        checkpoint_layout = QGridLayout(checkpoint_group)

        self.checkpoint_enabled = QCheckBox("Enable Checkpointing")
        self.checkpoint_enabled.setChecked(self.options.get("checkpoint_enabled", False))
        checkpoint_layout.addWidget(self.checkpoint_enabled, 0, 0, 1, 2)

        checkpoint_layout.addWidget(QLabel("Min Accuracy Increase %:"), 1, 0)
        self.checkpoint_threshold_spin = QDoubleSpinBox()
        self.checkpoint_threshold_spin.setRange(0.0, 10.0)
        self.checkpoint_threshold_spin.setDecimals(2)
        self.checkpoint_threshold_spin.setSingleStep(0.1)
        self.checkpoint_threshold_spin.setValue(self.options.get("checkpoint_threshold", 0.5))
        self.checkpoint_threshold_spin.setToolTip("Minimum accuracy increase (in %) to save a checkpoint")
        checkpoint_layout.addWidget(self.checkpoint_threshold_spin, 1, 1)

        checkpoint_layout.addWidget(QLabel("Save Directory:"), 2, 0)
        self.checkpoint_dir_button = QPushButton("Select Directory...")
        self.checkpoint_dir_button.clicked.connect(self.select_checkpoint_dir)
        checkpoint_layout.addWidget(self.checkpoint_dir_button, 2, 1)
        
        self.checkpoint_dir_label = QLabel(self.options.get("checkpoint_dir", "Not selected"))
        self.checkpoint_dir_label.setWordWrap(True)
        checkpoint_layout.addWidget(self.checkpoint_dir_label, 3, 0, 1, 2)

        self.checkpoint_enabled.toggled.connect(self.toggle_checkpoint_options)
        self.toggle_checkpoint_options()

        self.layout.addWidget(checkpoint_group, 3, 0, 1, 2)

        # --- Metric Tracking Section ---
        tracking_group = QGroupBox("Multi-Metric Tracking")
        tracking_layout = QGridLayout(tracking_group)

        self.tracking_enabled = QCheckBox("Enable Multi-Metric Tracking")
        self.tracking_enabled.setChecked(self.options.get("tracking_enabled", False))
        tracking_layout.addWidget(self.tracking_enabled, 0, 0, 1, 2)

        tracking_layout.addWidget(QLabel("Track Every N Epochs:"), 1, 0)
        self.tracking_frequency_spin = QSpinBox()
        self.tracking_frequency_spin.setRange(1, 100)
        self.tracking_frequency_spin.setValue(self.options.get("tracking_frequency", 1))
        self.tracking_frequency_spin.setToolTip("Calculate and log all selected metrics every N epochs")
        tracking_layout.addWidget(self.tracking_frequency_spin, 1, 1)

        tracking_layout.addWidget(QLabel("Metrics to Track:"), 2, 0, 1, 2)
        
        # Create checkboxes for each metric
        self.tracking_metric_checks = {}
        if TORCHMETRICS_AVAILABLE:
            available_metrics = [
                "Accuracy", "Precision", "Recall", "F1-Score", "AUROC", 
                "AveragePrecision", "Specificity", "MatthewsCorrCoef"
            ]
        else:
            available_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        
        row = 3
        col = 0
        for metric in available_metrics:
            check = QCheckBox(metric)
            check.setChecked(metric in self.options.get("tracked_metrics", []))
            self.tracking_metric_checks[metric] = check
            tracking_layout.addWidget(check, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        # Add beta parameter for F-Score tracking
        self.tracking_beta_label = QLabel("Tracking F-Score Beta:")
        self.tracking_beta_spin = QDoubleSpinBox()
        self.tracking_beta_spin.setRange(0.1, 10.0)
        self.tracking_beta_spin.setDecimals(1)
        self.tracking_beta_spin.setSingleStep(0.1)
        self.tracking_beta_spin.setValue(self.options.get("tracking_fscore_beta", 1.0))
        self.tracking_beta_spin.setToolTip("Beta parameter for tracked F-Score (beta=1 is F1, beta=2 is F2, etc.)")
        tracking_layout.addWidget(self.tracking_beta_label, row, 0)
        tracking_layout.addWidget(self.tracking_beta_spin, row, 1)

        self.tracking_enabled.toggled.connect(self.toggle_tracking_options)
        if "F1-Score" in self.tracking_metric_checks:
            self.tracking_metric_checks["F1-Score"].toggled.connect(self.toggle_tracking_beta)
        self.toggle_tracking_options()
        self.toggle_tracking_beta()

        self.layout.addWidget(tracking_group, 4, 0, 1, 2)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons, 5, 0, 1, 2)

        self.solver_combo.currentTextChanged.connect(self.toggle_options)
        self.scheduler_combo.currentTextChanged.connect(self.toggle_options)
        self.metric_combo.currentTextChanged.connect(self.toggle_options)
        self.toggle_options()

    def toggle_options(self):
        """Show/hide options based on selections."""
        solver = self.solver_combo.currentText()
        scheduler = self.scheduler_combo.currentText()

        # Optimizer specific
        self.momentum_label.setVisible(solver == "sgdm")
        self.momentum_spin.setVisible(solver == "sgdm")
        self.weight_decay_label.setVisible(solver in ["adam", "adamw", "sgdm"])
        self.weight_decay_spin.setVisible(solver in ["adam", "adamw", "sgdm"])

        # Scheduler specific
        is_step_or_plateau = scheduler in ["StepLR", "ReduceLROnPlateau"]
        self.lr_drop_factor_label.setVisible(is_step_or_plateau)
        self.lr_drop_factor_spin.setVisible(is_step_or_plateau)
        self.lr_drop_period_label.setVisible(is_step_or_plateau)
        self.lr_drop_period_spin.setVisible(is_step_or_plateau)
        
        if scheduler == "StepLR":
            self.lr_drop_period_label.setText("LR Drop Period (Epochs):")
        elif scheduler == "ReduceLROnPlateau":
            self.lr_drop_period_label.setText("LR Patience (Epochs):")

        # Metric specific
        is_fscore = self.metric_combo.currentText() == "F1-Score"
        self.beta_label.setVisible(is_fscore)
        self.beta_spin.setVisible(is_fscore)

    def toggle_checkpoint_options(self):
        """Enable/disable checkpoint-related widgets."""
        enabled = self.checkpoint_enabled.isChecked()
        self.checkpoint_threshold_spin.setEnabled(enabled)
        self.checkpoint_dir_button.setEnabled(enabled)

    def toggle_tracking_options(self):
        """Enable/disable tracking-related widgets."""
        enabled = self.tracking_enabled.isChecked()
        self.tracking_frequency_spin.setEnabled(enabled)
        for check in self.tracking_metric_checks.values():
            check.setEnabled(enabled)
        self.toggle_tracking_beta()

    def toggle_tracking_beta(self):
        """Show/hide tracking beta parameter based on F1-Score checkbox."""
        show = (self.tracking_enabled.isChecked() and 
                "F1-Score" in self.tracking_metric_checks and 
                self.tracking_metric_checks["F1-Score"].isChecked())
        self.tracking_beta_label.setVisible(show)
        self.tracking_beta_spin.setVisible(show)

    def select_checkpoint_dir(self):
        """Open a dialog to select the checkpoint directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory")
        if directory:
            self.options["checkpoint_dir"] = directory
            self.checkpoint_dir_label.setText(directory)

    def get_options(self):
        """Return the selected options as a dictionary."""
        self.options["solver"] = self.solver_combo.currentText()
        if self.options["solver"] == "sgdm":
            self.options["momentum"] = self.momentum_spin.value()
        if self.options["solver"] in ["adam", "adamw", "sgdm"]:
            self.options["weight_decay"] = self.weight_decay_spin.value()
        
        self.options["loss"] = self.loss_combo.currentText()
        self.options["scheduler"] = self.scheduler_combo.currentText()
        if self.options["scheduler"] in ["StepLR", "ReduceLROnPlateau"]:
            self.options["lr_drop_factor"] = self.lr_drop_factor_spin.value()
            self.options["lr_drop_period"] = self.lr_drop_period_spin.value()

        self.options["patience"] = self.patience_spin.value()
        self.options["shuffle"] = self.shuffle_combo.currentText()
        self.options["metric"] = self.metric_combo.currentText()
        self.options["fscore_beta"] = self.beta_spin.value()
        self.options["hflip"] = self.hflip_check.isChecked()
        self.options["vflip"] = self.vflip_check.isChecked()
        self.options["rotation"] = self.rotation_check.isChecked()
        self.options["class_balance_ratio"] = self.class_balance_spin.value()
        self.options["checkpoint_enabled"] = self.checkpoint_enabled.isChecked()
        self.options["checkpoint_threshold"] = self.checkpoint_threshold_spin.value()
        if "checkpoint_dir" not in self.options:
            self.options["checkpoint_dir"] = ""
        
        self.options["tracking_enabled"] = self.tracking_enabled.isChecked()
        self.options["tracking_frequency"] = self.tracking_frequency_spin.value()
        self.options["tracked_metrics"] = [metric for metric, check in self.tracking_metric_checks.items() if check.isChecked()]
        self.options["tracking_fscore_beta"] = self.tracking_beta_spin.value()
        
        return self.options

    @staticmethod
    def get_default_options():
        return {
            "solver": "adam",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "loss": "CrossEntropyLoss",
            "scheduler": "none",
            "lr_drop_factor": 0.1,
            "lr_drop_period": 10,
            "patience": 0,
            "shuffle": "every-epoch",
            "metric": "Accuracy",
            "fscore_beta": 1.0,
            "hflip": False,
            "vflip": False,
            "rotation": False,
            "class_balance_ratio": 0.0,
            "checkpoint_enabled": False,
            "checkpoint_threshold": 0.5,
            "checkpoint_dir": "",
            "tracking_enabled": False,
            "tracking_frequency": 1,
            "tracked_metrics": [],
            "tracking_fscore_beta": 1.0,
        }


class TrainingThread(QThread):
    """
    Runs the PyTorch training loop in a separate thread to keep the GUI responsive.
    """
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    epoch_result = pyqtSignal(dict)
    finished = pyqtSignal(object, dict)

    def __init__(self, data_path, model_name, epochs, lr, batch_size, splits, data_subset, advanced_options, pretrained_model_path=""):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.splits = splits # (train, val, test)
        self.data_subset = data_subset
        self.advanced_options = advanced_options
        self.pretrained_model_path = pretrained_model_path
        self.is_running = True
        
        # Initialize metric tracking history
        self.metric_history = {
            "train": {},
            "val": {},
            "epochs": []
        }

    def stop(self):
        self.is_running = False
        self.log.emit("Training termination requested...")

    def calculate_metric(self, all_preds, all_labels, metric_name, beta=1.0, num_classes=None, all_outputs=None):
        """Calculate the specified metric from predictions and labels using torchmetrics.
        
        Args:
            all_preds: List of prediction tensors (class indices)
            all_labels: List of label tensors
            metric_name: Name of the metric to calculate
            beta: Beta parameter for F-Score (default 1.0 for F1-Score)
            num_classes: Number of classes for the task
            all_outputs: List of raw output tensors (logits/probabilities) for metrics like AUROC
        """
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if num_classes is None:
            num_classes = torch.unique(all_labels).numel()
        
        task = "multiclass" if num_classes > 2 else "binary"
        
        # For probability-based metrics, use raw outputs if available
        needs_probabilities = metric_name in ["AUROC", "AveragePrecision"]
        
        try:
            if TORCHMETRICS_AVAILABLE:
                # Use torchmetrics for calculation
                if metric_name == "Accuracy":
                    metric = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
                elif metric_name == "Precision":
                    metric = torchmetrics.Precision(task=task, num_classes=num_classes, average="macro")
                elif metric_name == "Recall":
                    metric = torchmetrics.Recall(task=task, num_classes=num_classes, average="macro")
                elif metric_name == "F1-Score":
                    metric = torchmetrics.FBetaScore(task=task, num_classes=num_classes, beta=beta, average="macro")
                elif metric_name == "AUROC":
                    if all_outputs is None:
                        self.log.emit(f"Warning: AUROC requires raw outputs, using Accuracy instead.")
                        metric = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
                        result = metric(all_preds, all_labels)
                        return 100.0 * result.item()
                    metric = torchmetrics.AUROC(task=task, num_classes=num_classes, average="macro")
                    all_outputs_cat = torch.cat(all_outputs)
                    # Apply softmax to get probabilities
                    if task == "multiclass":
                        all_outputs_cat = torch.softmax(all_outputs_cat, dim=1)
                    result = metric(all_outputs_cat, all_labels)
                    return 100.0 * result.item()
                elif metric_name == "AveragePrecision":
                    if all_outputs is None:
                        self.log.emit(f"Warning: AveragePrecision requires raw outputs, using Accuracy instead.")
                        metric = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
                        result = metric(all_preds, all_labels)
                        return 100.0 * result.item()
                    metric = torchmetrics.AveragePrecision(task=task, num_classes=num_classes, average="macro")
                    all_outputs_cat = torch.cat(all_outputs)
                    # Apply softmax to get probabilities
                    if task == "multiclass":
                        all_outputs_cat = torch.softmax(all_outputs_cat, dim=1)
                    result = metric(all_outputs_cat, all_labels)
                    return 100.0 * result.item()
                elif metric_name == "Specificity":
                    metric = torchmetrics.Specificity(task=task, num_classes=num_classes, average="macro")
                elif metric_name == "MatthewsCorrCoef":
                    metric = torchmetrics.MatthewsCorrCoef(task=task, num_classes=num_classes)
                elif metric_name == "CohenKappa":
                    metric = torchmetrics.CohenKappa(task=task, num_classes=num_classes)
                elif metric_name == "JaccardIndex":
                    metric = torchmetrics.JaccardIndex(task=task, num_classes=num_classes, average="macro")
                elif metric_name == "Dice":
                    metric = torchmetrics.Dice(num_classes=num_classes, average="macro")
                elif metric_name == "HammingDistance":
                    metric = torchmetrics.HammingDistance(task=task, num_classes=num_classes)
                else:
                    # Fallback to accuracy
                    metric = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
                
                result = metric(all_preds, all_labels)
                return 100.0 * result.item()
            else:
                # Fallback to basic accuracy calculation if torchmetrics not available
                correct = (all_preds == all_labels).sum().item()
                return 100.0 * correct / len(all_labels)
        except Exception as e:
            self.log.emit(f"Warning: Error calculating {metric_name}: {e}. Using accuracy instead.")
            correct = (all_preds == all_labels).sum().item()
            return 100.0 * correct / len(all_labels)

    def balance_dataset(self, dataset, max_ratio):
        """Balance dataset classes by undersampling to meet the maximum ratio constraint.
        
        Args:
            dataset: ImageFolder dataset
            max_ratio: Maximum allowed ratio between largest and smallest class
            
        Returns:
            Balanced dataset (Subset)
        """
        from collections import Counter
        
        # Get class distribution
        targets = [label for _, label in dataset.samples]
        class_counts = Counter(targets)
        
        # Find min and max class sizes
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        # Calculate target max size based on ratio
        target_max = int(min_count * max_ratio)
        
        # Log current distribution
        self.log.emit(f"  Current class distribution: {dict(class_counts)}")
        self.log.emit(f"  Min class size: {min_count}, Max class size: {max_count}, Ratio: {max_count/min_count:.2f}")
        
        if max_count / min_count <= max_ratio:
            self.log.emit(f"  Dataset already balanced within ratio {max_ratio}")
            return dataset
        
        # Select indices for each class
        balanced_indices = []
        for class_idx in range(len(dataset.classes)):
            # Get all indices for this class
            class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
            
            # Limit to target_max samples
            if len(class_indices) > target_max:
                # Random undersample
                import random
                random.seed(42)  # For reproducibility
                class_indices = random.sample(class_indices, target_max)
            
            balanced_indices.extend(class_indices)
        
        # Create subset with balanced indices
        balanced_dataset = torch.utils.data.Subset(dataset, balanced_indices)
        
        # Log new distribution
        new_targets = [dataset.samples[i][1] for i in balanced_indices]
        new_counts = Counter(new_targets)
        new_min = min(new_counts.values())
        new_max = max(new_counts.values())
        self.log.emit(f"  New class distribution: {dict(new_counts)}")
        self.log.emit(f"  New ratio: {new_max/new_min:.2f}")
        
        return balanced_dataset

    def run(self):
        try:
            if not TORCH_AVAILABLE:
                self.log.emit(f"PyTorch is not available: {TORCH_ERROR}")
                self.finished.emit(None)
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log.emit(f"Using device: {device}")

            # Data transformations
            transforms_list = [
                transforms.Resize((224, 224)),
            ]
            if self.advanced_options.get("hflip", False):
                transforms_list.append(transforms.RandomHorizontalFlip())
                self.log.emit("Using random horizontal flips.")
            if self.advanced_options.get("vflip", False):
                transforms_list.append(transforms.RandomVerticalFlip())
                self.log.emit("Using random vertical flips.")
            if self.advanced_options.get("rotation", False):
                transforms_list.append(transforms.RandomRotation(10))
                self.log.emit("Using random rotations.")
            
            transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform = transforms.Compose(transforms_list)

            # Load dataset
            full_dataset = datasets.ImageFolder(self.data_path, transform=transform)
            num_classes = len(full_dataset.classes)
            class_names = full_dataset.classes
            self.log.emit(f"Found {len(full_dataset)} images in {num_classes} classes: {', '.join(class_names)}")

            # Apply class balancing if requested
            class_balance_ratio = self.advanced_options.get("class_balance_ratio", 0.0)
            if class_balance_ratio > 1.0:
                self.log.emit(f"Applying class balancing with max ratio: {class_balance_ratio}")
                full_dataset = self.balance_dataset(full_dataset, class_balance_ratio)
                self.log.emit(f"After balancing: {len(full_dataset)} images")

            # Subset the dataset if requested
            if self.data_subset != 'all':
                subset_size = int(self.data_subset)
                if subset_size < len(full_dataset):
                    self.log.emit(f"Using a random subset of {subset_size} images from the dataset.")
                    indices = torch.randperm(len(full_dataset)).tolist()
                    full_dataset = torch.utils.data.Subset(full_dataset, indices[:subset_size])
                else:
                    self.log.emit(f"Requested subset size ({subset_size}) is larger than or equal to dataset size ({len(full_dataset)}). Using full dataset.")

            # Split dataset
            train_prop, val_prop, test_prop = self.splits
            total_size = len(full_dataset)
            train_size = int(train_prop * total_size)
            val_size = int(val_prop * total_size)
            test_size = total_size - train_size - val_size
            
            if train_size + val_size + test_size > total_size:
                # Adjust to prevent overallocation from rounding
                train_size -= (train_size + val_size + test_size) - total_size

            self.log.emit(f"Splitting data: {train_size} train, {val_size} validation, {test_size} test.")
            
            # Ensure we have at least one sample in each split if proportions are non-zero
            if (train_prop > 0 and train_size == 0) or \
               (val_prop > 0 and val_size == 0) or \
               (test_prop > 0 and test_size == 0 and test_prop < 1.0): # test_size can be 0 if test_prop is 0
                self.log.emit("Warning: Dataset is too small for the given split proportions. Adjusting to ensure each split has at least one sample.")
                # A simple fallback, could be made more sophisticated
                if train_prop > 0: train_size = max(1, train_size)
                if val_prop > 0: val_size = max(1, val_size)
                if test_prop > 0: test_size = max(1, test_size)
                # This might still fail if total_size < number of non-zero splits, but it's a start.

            # Check if any split has zero samples when it shouldn't
            if train_size <= 0 or val_size <= 0:
                self.log.emit("Error: Training and validation sets must not be empty. Adjust split proportions or add more data.")
                self.finished.emit(None)
                return

            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) if test_size > 0 else None

            # Get model
            model = self.get_model(self.model_name, num_classes)
            if model is None:
                self.finished.emit(None)
                return
            
            # Load pretrained weights if provided
            if self.pretrained_model_path and os.path.exists(self.pretrained_model_path):
                try:
                    state_dict = torch.load(self.pretrained_model_path, map_location=device)
                    model.load_state_dict(state_dict, strict=False)
                    self.log.emit(f"Loaded pretrained weights from: {self.pretrained_model_path}")
                    self.log.emit("Continuing training from the loaded checkpoint.")
                except Exception as e:
                    self.log.emit(f"Warning: Failed to load pretrained weights: {e}")
                    self.log.emit("Proceeding with freshly initialized weights.")
            
            model = model.to(device)

            # --- Setup from Advanced Options ---
            # Loss Function
            loss_name = self.advanced_options.get("loss", "CrossEntropyLoss")
            if loss_name == "CrossEntropyLoss":
                criterion = nn.CrossEntropyLoss()
            elif loss_name == "NLLLoss":
                criterion = nn.NLLLoss()
            elif loss_name == "BCELoss":
                criterion = nn.BCELoss()
            else:
                self.log.emit(f"Warning: Unsupported loss function '{loss_name}'. Defaulting to CrossEntropyLoss.")
                criterion = nn.CrossEntropyLoss()
            self.log.emit(f"Using loss function: {loss_name}")

            # Optimizer
            solver_name = self.advanced_options.get("solver", "adam")
            weight_decay = self.advanced_options.get("weight_decay", 0.0)
            
            if solver_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=weight_decay)
            elif solver_name == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=weight_decay)
            elif solver_name == "sgdm":
                momentum = self.advanced_options.get("momentum", 0.9)
                optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)
            elif solver_name == "rmsprop":
                optimizer = optim.RMSprop(model.parameters(), lr=self.lr)
            elif solver_name == "adagrad":
                optimizer = optim.Adagrad(model.parameters(), lr=self.lr, weight_decay=weight_decay)
            else:
                self.log.emit(f"Warning: Unsupported optimizer '{solver_name}'. Defaulting to Adam.")
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.log.emit(f"Using optimizer: {solver_name}")

            # LR Scheduler
            scheduler_name = self.advanced_options.get("scheduler", "none")
            scheduler = None
            if scheduler_name == "StepLR":
                drop_period = self.advanced_options.get("lr_drop_period", 10)
                drop_factor = self.advanced_options.get("lr_drop_factor", 0.1)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=drop_period, gamma=drop_factor)
                self.log.emit(f"Using StepLR scheduler: drops by {drop_factor} every {drop_period} epochs.")
            elif scheduler_name == "ReduceLROnPlateau":
                patience = self.advanced_options.get("lr_drop_period", 10)
                factor = self.advanced_options.get("lr_drop_factor", 0.1)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)
                self.log.emit(f"Using ReduceLROnPlateau scheduler: reduces LR on validation loss plateau.")
            elif scheduler_name == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
                self.log.emit("Using CosineAnnealingLR scheduler.")
            
            # --- End Advanced Options Setup ---

            best_val_acc = 0.0
            best_model_state = None
            epochs_no_improve = 0
            patience = self.advanced_options.get("patience", 0)
            last_checkpoint_acc = 0.0  # Track the last accuracy at which we saved a checkpoint

            # Training loop
            for epoch in range(self.epochs):
                if not self.is_running:
                    break

                # Training phase
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    if not self.is_running:
                        break
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(train_loader.dataset)

                # Validation phase
                model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                all_outputs = []  # Store raw outputs for AUROC/AveragePrecision
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        all_preds.append(predicted.cpu())
                        all_labels.append(labels.cpu())
                        all_outputs.append(outputs.cpu())  # Store raw outputs

                val_loss /= len(val_loader.dataset)
                
                # Calculate the selected metric
                metric_name = self.advanced_options.get("metric", "Accuracy")
                beta = self.advanced_options.get("fscore_beta", 1.0)
                val_metric = self.calculate_metric(all_preds, all_labels, metric_name, beta, num_classes, all_outputs)

                # Multi-metric tracking
                if self.advanced_options.get("tracking_enabled", False):
                    tracking_freq = self.advanced_options.get("tracking_frequency", 1)
                    if (epoch + 1) % tracking_freq == 0:
                        tracked_metrics = self.advanced_options.get("tracked_metrics", [])
                        if tracked_metrics:
                            self.metric_history["epochs"].append(epoch + 1)
                            
                            # Store tracking beta value if F1-Score is being tracked
                            if "F1-Score" in tracked_metrics and "tracking_beta" not in self.metric_history:
                                self.metric_history["tracking_beta"] = self.advanced_options.get("tracking_fscore_beta", 1.0)
                            
                            # Calculate training metrics
                            train_preds_for_metrics = []
                            train_labels_for_metrics = []
                            train_outputs_for_metrics = []  # Store raw outputs
                            model.eval()
                            with torch.no_grad():
                                for inputs, labels in train_loader:
                                    inputs = inputs.to(device)
                                    outputs = model(inputs)
                                    _, predicted = torch.max(outputs.data, 1)
                                    train_preds_for_metrics.append(predicted.cpu())
                                    train_labels_for_metrics.append(labels)
                                    train_outputs_for_metrics.append(outputs.cpu())  # Store raw outputs
                            
                            for tracked_metric in tracked_metrics:
                                # Use tracking beta for F-Score, otherwise use primary beta
                                metric_beta = self.advanced_options.get("tracking_fscore_beta", 1.0) if tracked_metric == "F1-Score" else beta
                                
                                # Calculate for train set
                                train_score = self.calculate_metric(
                                    train_preds_for_metrics, 
                                    train_labels_for_metrics, 
                                    tracked_metric, 
                                    metric_beta, 
                                    num_classes,
                                    train_outputs_for_metrics
                                )
                                
                                # Calculate for val set
                                val_score = self.calculate_metric(
                                    all_preds, 
                                    all_labels, 
                                    tracked_metric, 
                                    metric_beta, 
                                    num_classes,
                                    all_outputs
                                )
                                
                                # Store in history
                                if tracked_metric not in self.metric_history["train"]:
                                    self.metric_history["train"][tracked_metric] = []
                                    self.metric_history["val"][tracked_metric] = []
                                
                                self.metric_history["train"][tracked_metric].append(train_score)
                                self.metric_history["val"][tracked_metric].append(val_score)
                            
                            self.log.emit(f"Tracked {len(tracked_metrics)} additional metrics at epoch {epoch+1}")

                # Format metric name for display
                display_metric_name = metric_name
                if metric_name == "F1-Score":
                    if beta == 1.0:
                        display_metric_name = "F1-Score"
                    else:
                        display_metric_name = f"F{beta:.1f}-Score"

                self.log.emit(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val {display_metric_name}: {val_metric:.2f}%")
                self.progress.emit(int(((epoch + 1) / self.epochs) * 100))

                if val_metric > best_val_acc:
                    best_val_acc = val_metric
                    best_model_state = model.state_dict()
                    epochs_no_improve = 0
                    
                    # Checkpointing logic
                    if self.advanced_options.get("checkpoint_enabled", False):
                        checkpoint_threshold = self.advanced_options.get("checkpoint_threshold", 0.5)
                        checkpoint_dir = self.advanced_options.get("checkpoint_dir", "")
                        
                        if checkpoint_dir and os.path.exists(checkpoint_dir):
                            metric_increase = val_metric - last_checkpoint_acc
                            if metric_increase >= checkpoint_threshold:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                # Format metric name for filename
                                file_metric_name = metric_name.lower().replace("-", "")
                                if metric_name == "F1-Score" and beta != 1.0:
                                    file_metric_name = f"f{beta:.1f}score".replace(".", "_")
                                
                                checkpoint_filename = f"{self.model_name}_epoch{epoch+1}_{file_metric_name}{val_metric:.2f}_{timestamp}.pth"
                                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                                try:
                                    torch.save(model.state_dict(), checkpoint_path)
                                    self.log.emit(f"Checkpoint saved: {checkpoint_filename} ({display_metric_name} improved by {metric_increase:.2f}%)")
                                    last_checkpoint_acc = val_metric
                                except Exception as e:
                                    self.log.emit(f"Failed to save checkpoint: {e}")
                        elif self.advanced_options.get("checkpoint_enabled", False):
                            self.log.emit("Warning: Checkpointing is enabled but no valid directory is set.")
                else:
                    epochs_no_improve += 1

                # LR scheduler step
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # Early stopping
                if patience > 0 and epochs_no_improve >= patience:
                    self.log.emit(f"Early stopping triggered after {patience} epochs with no improvement.")
                    break

            if best_model_state:
                model.load_state_dict(best_model_state)
                metric_name = self.advanced_options.get("metric", "Accuracy")
                beta = self.advanced_options.get("fscore_beta", 1.0)
                
                # Format metric name for display
                display_metric_name = metric_name
                if metric_name == "F1-Score":
                    if beta == 1.0:
                        display_metric_name = "F1-Score"
                    else:
                        display_metric_name = f"F{beta:.1f}-Score"
                
                self.log.emit(f"Training finished. Best validation {display_metric_name}: {best_val_acc:.2f}%")
                
                # Prepare metric info for export
                metric_info = {
                    "metric_name": metric_name,
                    "metric_value": best_val_acc,
                    "beta": beta,
                    "model_name": self.model_name,
                    "metric_history": self.metric_history
                }
                
                # Final test set evaluation
                if test_loader:
                    self.log.emit("Evaluating on the test set...")
                    model.eval()
                    test_loss = 0.0
                    all_preds = []
                    all_labels = []
                    all_outputs = []  # Store raw outputs
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            test_loss += loss.item() * inputs.size(0)
                            _, predicted = torch.max(outputs.data, 1)
                            all_preds.append(predicted.cpu())
                            all_labels.append(labels.cpu())
                            all_outputs.append(outputs.cpu())  # Store raw outputs
                    
                    test_loss /= len(test_loader.dataset)
                    beta = self.advanced_options.get("fscore_beta", 1.0)
                    test_metric = self.calculate_metric(all_preds, all_labels, metric_name, beta, num_classes, all_outputs)
                    
                    # Calculate confusion matrix
                    all_preds_cat = torch.cat(all_preds)
                    all_labels_cat = torch.cat(all_labels)
                    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
                    for t, p in zip(all_labels_cat, all_preds_cat):
                        confusion_matrix[t.long(), p.long()] += 1
                    
                    # Add confusion matrix to metric_info
                    metric_info["confusion_matrix"] = confusion_matrix.tolist()
                    metric_info["class_names"] = class_names
                    
                    # Format metric name for display
                    display_metric_name = metric_name
                    if metric_name == "F1-Score":
                        if beta == 1.0:
                            display_metric_name = "F1-Score"
                        else:
                            display_metric_name = f"F{beta:.1f}-Score"
                    
                    self.log.emit(f"Test Results - Loss: {test_loss:.4f}, {display_metric_name}: {test_metric:.2f}%")

                self.finished.emit(model, metric_info)
            else:
                self.log.emit("Training stopped or completed without improvement.")
                self.finished.emit(None, {})

        except Exception as e:
            self.log.emit(f"An error occurred during training: {e}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(None, {})

    def get_model(self, model_name, num_classes):
        """Initializes a pretrained model and replaces the final layer."""
        self.log.emit(f"Loading model: {model_name}... (Download progress will appear in terminal)")
        try:
            # Dynamically get the model constructor from torchvision.models
            model_constructor = getattr(models, model_name.lower())
            # The progress bar will now safely print to the original console
            model = model_constructor(weights='IMAGENET1K_V1', progress=True)
            self.log.emit(f"Successfully loaded pretrained weights for {model_name}.")

            # Generic way to replace the classifier
            if hasattr(model, 'fc'): # ResNet, GoogLeNet, Inception
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            elif hasattr(model, 'classifier'): # VGG, AlexNet, MobileNet, DenseNet, etc.
                if isinstance(model.classifier, nn.Linear):
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, num_classes)
                elif isinstance(model.classifier, nn.Sequential):
                    # Find the last Linear layer in the Sequential classifier
                    last_layer_idx = -1
                    for i, layer in reversed(list(enumerate(model.classifier))):
                        if isinstance(layer, nn.Linear):
                            last_layer_idx = i
                            break
                    
                    if last_layer_idx != -1:
                        num_ftrs = model.classifier[last_layer_idx].in_features
                        model.classifier[last_layer_idx] = nn.Linear(num_ftrs, num_classes)
                    else:
                        self.log.emit(f"Error: Could not find a Linear layer in the classifier of {model_name}.")
                        return None
                else:
                    self.log.emit(f"Error: Unsupported classifier type in {model_name}.")
                    return None
            else:
                self.log.emit(f"Error: Cannot find a classifier layer to replace for model {model_name}.")
                return None
            
            self.log.emit(f"Final layer of {model_name} adapted for {num_classes} classes.")
            return model

        except Exception as e:
            self.log.emit(f"Error loading model {model_name}: {e}")
            return None


class ChipTrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_path = ""
        self.loaded_model_path = ""
        self.training_thread = None
        self.trained_model = None
        self.best_metric_name = ""
        self.best_metric_value = 0.0
        self.model_architecture = ""
        self.metric_history = {}
        self.confusion_matrix = None
        self.class_names = []
        self.advanced_options = AdvancedOptionsDialog.get_default_options()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("ChipTrainer - PyTorch GUI")
        self.setGeometry(100, 100, 800, 750) # Increased size for better layout

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Data Selection ---
        data_group = QGroupBox("1. Data and Model")
        data_layout = QGridLayout(data_group)

        self.dir_button = QPushButton("Select Data Directory")
        self.dir_button.clicked.connect(self.select_directory)
        self.dir_label = QLabel("No directory selected.")
        data_layout.addWidget(self.dir_button, 0, 0)
        data_layout.addWidget(self.dir_label, 0, 1, 1, 3)

        data_layout.addWidget(QLabel("Model Architecture:"), 1, 0)
        self.model_combo = QComboBox()
        # Populate with a comprehensive list of torchvision models
        available_models = [
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'AlexNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19',
            'GoogLeNet', 'Inception_V3',
            'DenseNet121', 'DenseNet169', 'DenseNet201',
            'MobileNet_V2', 'MobileNet_V3_Large', 'MobileNet_V3_Small',
            'SqueezeNet1_0', 'SqueezeNet1_1',
            'ShuffleNet_V2_x0_5', 'ShuffleNet_V2_x1_0',
            'MNASNet0_5', 'MNASNet1_0',
            'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2', 'EfficientNet_B3',
            'EfficientNet_B4', 'EfficientNet_B5', 'EfficientNet_B6', 'EfficientNet_B7',
            'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_X_400MF', 'RegNet_X_800MF',
            'Wide_ResNet50_2', 'Wide_ResNet101_2'
        ]
        self.model_combo.addItems(available_models)
        data_layout.addWidget(self.model_combo, 1, 1, 1, 3)

        data_layout.addWidget(QLabel("Data Subset to Use:"), 2, 0)
        self.data_subset_input = QLineEdit("all")
        self.data_subset_input.setToolTip("Enter an integer number of images to use, or 'all' to use the entire dataset.")
        data_layout.addWidget(self.data_subset_input, 2, 1)

        self.load_model_button = QPushButton("Load Pre-trained Model...")
        self.load_model_button.clicked.connect(self.load_pretrained_model)
        data_layout.addWidget(self.load_model_button, 3, 0)
        
        self.load_best_button = QPushButton("Load Best from Directory...")
        self.load_best_button.clicked.connect(self.load_best_from_directory)
        self.load_best_button.setToolTip("Automatically find and load the best performing model from a directory")
        data_layout.addWidget(self.load_best_button, 3, 2)
        
        self.loaded_model_label = QLabel("No model loaded")
        data_layout.addWidget(self.loaded_model_label, 3, 1, 1, 1)
        
        # --- Data Split ---
        split_group = QGroupBox("2. Data Split Proportions")
        split_layout = QGridLayout(split_group)
        
        split_layout.addWidget(QLabel("Train:"), 0, 0)
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.0, 1.0)
        self.train_split_spin.setDecimals(2)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.7)
        split_layout.addWidget(self.train_split_spin, 0, 1)

        split_layout.addWidget(QLabel("Validation:"), 0, 2)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 1.0)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.2)
        split_layout.addWidget(self.val_split_spin, 0, 3)

        split_layout.addWidget(QLabel("Test:"), 0, 4)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.0, 1.0)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.1)
        split_layout.addWidget(self.test_split_spin, 0, 5)
        
        self.train_split_spin.valueChanged.connect(self.validate_splits)
        self.val_split_spin.valueChanged.connect(self.validate_splits)
        self.test_split_spin.valueChanged.connect(self.validate_splits)

        main_layout.addWidget(data_group)
        main_layout.addWidget(split_group)

        # --- Training Parameters ---
        params_group = QGroupBox("3. Training Parameters")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin, 0, 1)

        params_layout.addWidget(QLabel("Learning Rate:"), 0, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.001)
        params_layout.addWidget(self.lr_spin, 0, 3)

        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        params_layout.addWidget(self.batch_spin, 1, 1)

        self.advanced_button = QPushButton("Advanced Options...")
        self.advanced_button.clicked.connect(self.open_advanced_options)
        params_layout.addWidget(self.advanced_button, 1, 2, 1, 2)

        main_layout.addWidget(params_group)

        # --- Training Controls ---
        controls_group = QGroupBox("4. Execution")
        controls_layout = QHBoxLayout(controls_group)
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.export_button = QPushButton("Export Model")
        self.export_button.clicked.connect(self.export_model)
        self.export_button.setEnabled(False)
        
        self.export_metrics_button = QPushButton("Export Metrics")
        self.export_metrics_button.clicked.connect(self.export_metric_history)
        self.export_metrics_button.setEnabled(False)
        
        self.export_config_button = QPushButton("Export Config")
        self.export_config_button.clicked.connect(self.export_config)
        self.export_config_button.setToolTip("Export current settings to JSON configuration file")
        
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.export_metrics_button)
        controls_layout.addWidget(self.export_config_button)
        main_layout.addWidget(controls_group)

        # --- Log and Progress ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.progress_bar = QProgressBar()
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.progress_bar)
        main_layout.addWidget(log_group)

        # --- GPU Status ---
        if GPUTIL_AVAILABLE:
            self.gpu_group = QGroupBox("GPU Status")
            self.gpu_layout = QVBoxLayout(self.gpu_group) # Use QVBoxLayout for a single column
            main_layout.addWidget(self.gpu_group)

        # --- Status Bar ---
        self.statusBar().showMessage("Ready")
        if not TORCH_AVAILABLE:
            self.statusBar().showMessage(f"PyTorch Error: {TORCH_ERROR}")
            QMessageBox.critical(self, "Error", f"Failed to import PyTorch or Torchvision.\nPlease ensure they are installed.\n\n{TORCH_ERROR}")

        self.init_gpu_monitor()



    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Data Directory")
        if directory:
            try:
                subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
                if not subfolders:
                    QMessageBox.warning(self, "Warning", "The selected directory contains no subdirectories (classes).")
                    return
                self.data_path = directory
                self.dir_label.setText(Path(directory).name)
                self.log_text.append(f"Data directory set to: {directory}")
                self.log_text.append(f"Found {len(subfolders)} potential classes.")
                self.train_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not read the directory.\n{e}")

    def load_pretrained_model(self):
        """Load a pre-trained model from disk to continue training."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Model Weights", "", "PyTorch Model (*.pth *.pt)")
        if filename:
            try:
                # Verify the file exists and can be loaded
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Model file not found: {filename}")
                
                # Try to load it to verify it's a valid PyTorch state dict
                torch.load(filename)
                
                self.loaded_model_path = filename
                self.loaded_model_label.setText(Path(filename).name)
                self.log_text.append(f"Pre-trained model loaded: {filename}")
                self.log_text.append("This model will be used as the starting point for training.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model.\n{e}")
                self.loaded_model_path = ""
                self.loaded_model_label.setText("No model loaded")

    def load_best_from_directory(self):
        """Automatically find and load the best performing model from a directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Model Checkpoints")
        if not directory:
            return
        
        try:
            # Find all .pth and .pt files
            model_files = []
            for ext in ['*.pth', '*.pt']:
                model_files.extend(Path(directory).glob(ext))
            
            if not model_files:
                QMessageBox.warning(self, "No Models Found", "No .pth or .pt files found in the selected directory.")
                return
            
            # Parse filenames to extract metric scores
            # Expected format: ModelName_epochN_metricXX.XX_timestamp.pth
            best_model = None
            best_score = -1.0
            best_metric_name = None
            
            import re
            for model_file in model_files:
                filename = model_file.stem  # filename without extension
                
                # Try to extract metric score from various common patterns
                # Examples: acc95.50, f1-score87.30, f2_0score85.00, matthewscorrcoef91.27
                patterns = [
                    r'(f\d+[_.-]?\d*-?score)(\d+\.\d+)',  # f1score, f2_0score, f1-score, f2.0-score
                    r'(acc|accuracy|precision|recall|auroc|dice|jaccard|hamming)(\d+\.\d+)',
                    r'(matthewscorrcoef|cohenkappa|specificity|averageprecision)(\d+\.\d+)',
                    r'_([a-z]+)(\d+\.\d+)_\d{8}',  # any metric name before timestamp
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, filename, re.IGNORECASE)
                    if match:
                        metric_name, score_str = match.groups()
                        
                        try:
                            score = float(score_str)
                            if score > best_score:
                                best_score = score
                                best_model = model_file
                                best_metric_name = metric_name
                            break
                        except ValueError:
                            continue
            
            if best_model:
                # Verify the file can be loaded
                torch.load(str(best_model))
                
                self.loaded_model_path = str(best_model)
                self.loaded_model_label.setText(best_model.name)
                self.log_text.append(f"Best model found and loaded: {best_model.name}")
                self.log_text.append(f"Score: {best_score:.2f} ({best_metric_name})")
                self.log_text.append(f"Found {len(model_files)} total models in directory.")
                QMessageBox.information(self, "Success", 
                    f"Loaded best model: {best_model.name}\n"
                    f"Score: {best_score:.2f} ({best_metric_name})\n"
                    f"Out of {len(model_files)} models found.")
            else:
                QMessageBox.warning(self, "Parse Error", 
                    f"Found {len(model_files)} model files but couldn't parse metric scores from filenames.\n"
                    f"Expected format: ModelName_epochN_metricXX.XX_timestamp.pth")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load best model from directory.\n{e}")
            self.loaded_model_path = ""
            self.loaded_model_label.setText("No model loaded")

    def start_training(self):
        if not self.data_path:
            QMessageBox.warning(self, "Warning", "Please select a data directory first.")
            return

        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        splits = (self.train_split_spin.value(), self.val_split_spin.value(), self.test_split_spin.value())
        if abs(sum(splits) - 1.0) > 0.001:
            QMessageBox.warning(self, "Warning", "Split proportions sum to more than 1.0. Please adjust.")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return

        data_subset_text = self.data_subset_input.text().strip().lower()
        if data_subset_text != 'all':
            try:
                subset_val = int(data_subset_text)
                if subset_val <= 0:
                    raise ValueError("Subset must be a positive integer.")
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Data Subset must be a positive integer or 'all'.")
                self.train_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return
        
        self.training_thread = TrainingThread(
            data_path=self.data_path,
            model_name=self.model_combo.currentText(),
            epochs=self.epochs_spin.value(),
            lr=self.lr_spin.value(),
            batch_size=self.batch_spin.value(),
            splits=splits,
            data_subset=data_subset_text,
            advanced_options=self.advanced_options,
            pretrained_model_path=self.loaded_model_path
        )
        self.training_thread.log.connect(self.log_text.append)
        self.training_thread.progress.connect(self.progress_bar.setValue)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.stop_button.setEnabled(False)

    def training_finished(self, model, metric_info):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if model:
            self.trained_model = model
            # Store metric info for export
            self.best_metric_name = metric_info.get("metric_name", "")
            self.best_metric_value = metric_info.get("metric_value", 0.0)
            self.model_architecture = metric_info.get("model_name", "")
            self.metric_history = metric_info.get("metric_history", {})
            self.confusion_matrix = metric_info.get("confusion_matrix", None)
            self.class_names = metric_info.get("class_names", [])
            self.export_button.setEnabled(True)
            
            # Enable export metrics button if tracking was enabled
            if self.metric_history and self.metric_history.get("epochs"):
                self.export_metrics_button.setEnabled(True)
            
            self.statusBar().showMessage("Training complete. Model is ready to be exported.")
        else:
            self.statusBar().showMessage("Training stopped or failed.")
            # Reset metric info if training failed
            self.best_metric_name = ""
            self.best_metric_value = 0.0
            self.model_architecture = ""
            self.metric_history = {}
            self.confusion_matrix = None
            self.class_names = []

    def open_advanced_options(self):
        dialog = AdvancedOptionsDialog(self, self.advanced_options)
        if dialog.exec_():
            self.advanced_options = dialog.get_options()
            self.log_text.append("Advanced options updated.")

    def export_metric_history(self):
        """Export tracked metric history to a JSON file with optional matplotlib plots."""
        if not self.metric_history or not self.metric_history.get("epochs"):
            QMessageBox.information(self, "No Data", "No metric history available to export.")
            return
        
        # Ask user if they want to generate matplotlib figures
        reply = QMessageBox.question(
            self, 
            "Export Options", 
            "Do you want to generate matplotlib figures for each metric?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        generate_plots = (reply == QMessageBox.Yes)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"{self.model_architecture}_metrics_{timestamp}.json"
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Metric History",
                default_filename,
                "JSON Files (*.json)"
            )
            
            if filename:
                # Prepare export data
                export_data = self.metric_history.copy()
                if self.confusion_matrix is not None:
                    export_data["confusion_matrix"] = self.confusion_matrix
                    export_data["class_names"] = self.class_names
                
                # Save JSON data
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.log_text.append(f"Metric history exported to: {filename}")
                
                # Generate matplotlib figures if requested
                if generate_plots:
                    self.generate_metric_plots(filename)
                    if self.confusion_matrix is not None:
                        self.generate_confusion_matrix_plot(filename)
                
                QMessageBox.information(self, "Success", f"Metric history saved to {filename}")
        except Exception as e:
            self.log_text.append(f"Failed to export metric history: {e}")
            QMessageBox.warning(self, "Export Error", f"Failed to export metric history:\n{e}")

    def export_config(self):
        """Export current GUI settings to a JSON configuration file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"training_config_{timestamp}.json"
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Configuration",
                default_filename,
                "JSON Files (*.json)"
            )
            
            if not filename:
                return
            
            # Build configuration dictionary
            config = {
                # Required fields
                "data_path": self.data_path if self.data_path else "/path/to/your/dataset",
                "model_name": self.model_combo.currentText(),
                "epochs": self.epochs_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "batch_size": self.batch_spin.value(),
                
                # Data configuration
                "train_split": self.train_split_spin.value(),
                "val_split": self.val_split_spin.value(),
                "test_split": self.test_split_spin.value(),
                "data_subset": self.data_subset_input.text().strip(),
                
                # Model loading
                "pretrained_model_path": self.loaded_model_path,
                
                # Advanced options
                "advanced_options": self.advanced_options.copy(),
                
                # Output configuration
                "output_dir": "./output",
                "export_model": True,
                "export_metrics": True
            }
            
            # Save configuration
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.log_text.append(f"Configuration exported to: {filename}")
            QMessageBox.information(self, "Success", f"Configuration saved to:\n{filename}\n\nYou can use this file with:\npython train_from_config.py {filename}")
            
        except Exception as e:
            self.log_text.append(f"Failed to export configuration: {e}")
            QMessageBox.warning(self, "Export Error", f"Failed to export configuration:\n{e}")

    def generate_metric_plots(self, json_filename):
        """Generate matplotlib figures for each tracked metric."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            self.log_text.append("Warning: matplotlib not available for generating plots.")
            return
        
        try:
            epochs = self.metric_history.get("epochs", [])
            train_metrics = self.metric_history.get("train", {})
            val_metrics = self.metric_history.get("val", {})
            
            if not epochs:
                return
            
            # Get base filename without extension
            base_filename = json_filename.rsplit('.', 1)[0]
            
            # Create a plot for each metric
            for metric_name in train_metrics.keys():
                if metric_name not in val_metrics:
                    continue
                
                # Format display name for F-Score with correct beta
                display_name = metric_name
                if metric_name == "F1-Score":
                    tracking_beta = self.metric_history.get("tracking_beta", 1.0)
                    if tracking_beta == 1.0:
                        display_name = "F1-Score"
                    else:
                        display_name = f"F{tracking_beta:.1f}-Score"
                
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_metrics[metric_name], 'b-o', label='Train', linewidth=2, markersize=6)
                plt.plot(epochs, val_metrics[metric_name], 'r-s', label='Validation', linewidth=2, markersize=6)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(f'{display_name} (%)', fontsize=12)
                plt.title(f'{self.model_architecture} - {display_name} over Training', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                plot_filename = f"{base_filename}_{metric_name.replace('-', '').replace(' ', '_')}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log_text.append(f"Plot saved: {plot_filename}")
            
            self.log_text.append(f"Generated {len(train_metrics)} metric plots.")
            
        except Exception as e:
            self.log_text.append(f"Warning: Failed to generate plots: {e}")

    def generate_confusion_matrix_plot(self, json_filename):
        """Generate confusion matrix heatmap visualization."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            self.log_text.append("Warning: matplotlib/numpy not available for generating confusion matrix plot.")
            return
        
        try:
            if self.confusion_matrix is None or not self.class_names:
                return
            
            # Get base filename without extension
            base_filename = json_filename.rsplit('.', 1)[0]
            
            # Convert to numpy array
            cm = np.array(self.confusion_matrix)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(8, len(self.class_names)), max(8, len(self.class_names))))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Set ticks and labels
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   xlabel='Predicted Label',
                   ylabel='True Label',
                   title=f'{self.model_architecture} - Confusion Matrix (Test Set)')
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Loop over data dimensions and create text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            fig.tight_layout()
            
            # Save figure
            plot_filename = f"{base_filename}_confusion_matrix.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log_text.append(f"Confusion matrix plot saved: {plot_filename}")
            
        except Exception as e:
            self.log_text.append(f"Warning: Failed to generate confusion matrix plot: {e}")

    def validate_splits(self):
        total = self.train_split_spin.value() + self.val_split_spin.value() + self.test_split_spin.value()
        if abs(total - 1.0) > 0.001:
            self.statusBar().showMessage(f"Warning: Split proportions sum to {total:.2f}, not 1.0.", 5000)
        else:
            self.statusBar().clearMessage()

    def export_model(self):
        if not self.trained_model:
            QMessageBox.warning(self, "Warning", "No model has been trained yet.")
            return

        # Generate default filename in checkpoint format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.best_metric_name and self.model_architecture:
            # Format metric name for filename
            metric_name = self.best_metric_name
            file_metric_name = metric_name.lower().replace("-", "")
            
            # Handle F-Score with beta
            if metric_name == "F1-Score":
                beta = self.advanced_options.get("fscore_beta", 1.0)
                if beta != 1.0:
                    file_metric_name = f"f{beta:.1f}score".replace(".", "_")
            
            default_filename = f"{self.model_architecture}_final_{file_metric_name}{self.best_metric_value:.2f}_{timestamp}.pth"
        else:
            # Fallback if no metric info available
            default_filename = f"model_{timestamp}.pth"

        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Model", 
            default_filename, 
            "PyTorch Model (*.pth)"
        )
        if filename:
            try:
                torch.save(self.trained_model.state_dict(), filename)
                self.log_text.append(f"Model state_dict saved to: {filename}")
                QMessageBox.information(self, "Success", f"Model saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model.\n{e}")

    def init_gpu_monitor(self):
        if not GPUTIL_AVAILABLE:
            self.log_text.append("GPUtil not found. GPU monitoring is disabled.")
            return

        self.gpu_labels = {}
        labels_to_create = ["ID", "Name", "Load", "Free Memory", "Used Memory", "Total Memory", "Temperature"]
        for text in labels_to_create:
            label = QLabel(f"{text}: N/A")
            self.gpu_layout.addWidget(label)
            self.gpu_labels[text] = label

        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self.update_gpu_stats)
        self.gpu_timer.start(2000) # Update every 2 seconds
        self.update_gpu_stats() # Initial update

    def update_gpu_stats(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0] # Display info for the first GPU
                self.gpu_labels["ID"].setText(f"ID: {gpu.id}")
                self.gpu_labels["Name"].setText(f"Name: {gpu.name}")
                self.gpu_labels["Load"].setText(f"Load: {gpu.load*100:.1f}%")
                self.gpu_labels["Free Memory"].setText(f"Free Memory: {gpu.memoryFree:.1f}MB")
                self.gpu_labels["Used Memory"].setText(f"Used Memory: {gpu.memoryUsed:.1f}MB")
                self.gpu_labels["Total Memory"].setText(f"Total Memory: {gpu.memoryTotal:.1f}MB")
                self.gpu_labels["Temperature"].setText(f"Temperature: {gpu.temperature}°C")
            else:
                # Clear fields if no GPU is detected
                for key in self.gpu_labels:
                    self.gpu_labels[key].setText(f"{key}: N/A")
                self.gpu_labels["Name"].setText("Name: No GPU detected")
        except Exception as e:
            self.gpu_labels["Name"].setText(f"Name: Error updating GPU stats")
            self.gpu_timer.stop()

    def closeEvent(self, event):
        self.stop_training()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ChipTrainerGUI()
    gui.show()
    sys.exit(app.exec_())
