import sys
import os
import json
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import seaborn as sns

# Fix for PyTorch DLL loading issues on Windows with Qt
# Import PyTorch BEFORE PyQt to avoid DLL conflicts
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

# Now import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QGridLayout, QMessageBox, QComboBox,
                             QDialog, QButtonGroup, QRadioButton, QCheckBox,
                             QScrollArea, QFormLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
from pathlib import Path
from datetime import datetime

# Custom loss functions
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss to prevent overconfidence"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = (-targets_smooth * log_probs).sum(dim=-1).mean()
        return loss

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, data_path, epochs, learning_rate, batch_size, train_split, val_split, test_split, 
                 model_name, optimizer_name, scheduler_name, patience, augmentation_settings, 
                 loss_function, shuffle_frequency):
        super().__init__()
        self.data_path = data_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.patience = patience
        self.augmentation_settings = augmentation_settings
        self.loss_function = loss_function
        self.shuffle_frequency = shuffle_frequency
        # Periodic save settings (save every N epochs, 0 = disabled)
        self.save_every_n = 0
        self.save_output_dir = ""
        # Store trained model and metadata
        self.trained_model = None
        self.model_metadata = {}
        # Store predictions and labels for statistics
        self.test_predictions = []
        self.test_labels = []
        self.test_probabilities = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []
        
    def run(self):
        try:
            if not TORCH_AVAILABLE:
                self.log.emit(f"Error: PyTorch not available - {TORCH_ERROR}")
                return
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.log.emit(f"Starting training on device: {self.device} ({device_name})")
            else:
                self.log.emit(f"Starting training on device: {self.device}")
            self.log.emit(f"Data path: {self.data_path}")
            self.log.emit(f"Data split - Train: {self.train_split:.2f}, Val: {self.val_split:.2f}, Test: {self.test_split:.2f}")
            self.log.emit(f"Optimizer: {self.optimizer_name}, LR Scheduler: {self.scheduler_name}")
            if self.patience > 0:
                self.log.emit(f"Early stopping enabled with patience: {self.patience}")
            
            # Build data transforms with custom augmentation
            transform_list = []
            
            # Start with resize
            if 'RandomCrop' in self.augmentation_settings:
                crop_size = self.augmentation_settings['RandomCrop']['size']
                transform_list.append(transforms.Resize((int(crop_size * 1.14), int(crop_size * 1.14))))
            else:
                transform_list.append(transforms.Resize((224, 224)))
            
            # Add augmentations
            aug_names = []
            if 'RandomCrop' in self.augmentation_settings:
                size = self.augmentation_settings['RandomCrop']['size']
                transform_list.append(transforms.RandomCrop(size))
                aug_names.append(f"RandomCrop({size})")
            
            if 'RandomHorizontalFlip' in self.augmentation_settings:
                p = self.augmentation_settings['RandomHorizontalFlip']['p']
                transform_list.append(transforms.RandomHorizontalFlip(p=p))
                aug_names.append(f"RandomHorizontalFlip(p={p})")
            
            if 'RandomVerticalFlip' in self.augmentation_settings:
                p = self.augmentation_settings['RandomVerticalFlip']['p']
                transform_list.append(transforms.RandomVerticalFlip(p=p))
                aug_names.append(f"RandomVerticalFlip(p={p})")
            
            if 'RandomRotation' in self.augmentation_settings:
                degrees = self.augmentation_settings['RandomRotation']['degrees']
                transform_list.append(transforms.RandomRotation(degrees))
                aug_names.append(f"RandomRotation({degrees}°)")
            
            if 'RandomPerspective' in self.augmentation_settings:
                p = self.augmentation_settings['RandomPerspective']['p']
                transform_list.append(transforms.RandomPerspective(p=p))
                aug_names.append(f"RandomPerspective(p={p})")
            
            if 'ColorJitter' in self.augmentation_settings:
                settings = self.augmentation_settings['ColorJitter']
                transform_list.append(transforms.ColorJitter(
                    brightness=settings['brightness'],
                    contrast=settings['contrast'],
                    saturation=settings['saturation'],
                    hue=settings['hue']
                ))
                aug_names.append("ColorJitter")
            
            if 'RandomGrayscale' in self.augmentation_settings:
                p = self.augmentation_settings['RandomGrayscale']['p']
                transform_list.append(transforms.RandomGrayscale(p=p))
                aug_names.append(f"RandomGrayscale(p={p})")
            
            if 'GaussianBlur' in self.augmentation_settings:
                settings = self.augmentation_settings['GaussianBlur']
                kernel_size = settings['kernel_size']
                # Ensure kernel size is odd
                if kernel_size % 2 == 0:
                    kernel_size += 1
                transform_list.append(transforms.GaussianBlur(
                    kernel_size=kernel_size,
                    sigma=(settings['sigma_min'], settings['sigma_max'])
                ))
                aug_names.append("GaussianBlur")
            
            if 'RandomAdjustSharpness' in self.augmentation_settings:
                settings = self.augmentation_settings['RandomAdjustSharpness']
                transform_list.append(transforms.RandomAdjustSharpness(
                    sharpness_factor=settings['sharpness_factor'],
                    p=settings['p']
                ))
                aug_names.append("RandomAdjustSharpness")
            
            # Convert to tensor
            transform_list.append(transforms.ToTensor())
            
            # Add RandomErasing after ToTensor
            if 'RandomErasing' in self.augmentation_settings:
                settings = self.augmentation_settings['RandomErasing']
                transform_list.append(transforms.RandomErasing(
                    p=settings['p'],
                    scale=(settings['scale_min'], settings['scale_max'])
                ))
                aug_names.append("RandomErasing")
            
            # Normalize
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            
            train_transform = transforms.Compose(transform_list)
            
            if aug_names:
                self.log.emit(f"Using custom data augmentation: {', '.join(aug_names)}")
            else:
                self.log.emit("No data augmentation enabled")
            
            # Validation/test transform (no augmentation)
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Load datasets with appropriate transforms
            full_dataset = datasets.ImageFolder(self.data_path, transform=val_transform)
            
            num_classes = len(full_dataset.classes)
            self.log.emit(f"Found {num_classes} classes: {', '.join(full_dataset.classes)}")
            self.log.emit(f"Total images: {len(full_dataset)}")
            
            # Split dataset into train, validation, and test sets
            dataset_size = len(full_dataset)
            train_size = int(self.train_split * dataset_size)
            val_size = int(self.val_split * dataset_size)
            test_size = dataset_size - train_size - val_size  # Remaining goes to test
            
            train_indices, val_indices, test_indices = torch.utils.data.random_split(
                range(dataset_size), 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            # Create datasets with different transforms
            train_dataset = datasets.ImageFolder(self.data_path, transform=train_transform)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
            
            val_dataset = datasets.ImageFolder(self.data_path, transform=val_transform)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
            
            test_dataset = datasets.ImageFolder(self.data_path, transform=val_transform)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices.indices)
            
            self.log.emit(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            
            # Create dataloaders with shuffle control
            should_shuffle = self.shuffle_frequency > 0
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=should_shuffle)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            if self.shuffle_frequency > 0:
                self.log.emit(f"Training data will be reshuffled every {self.shuffle_frequency} epoch(s)")
            else:
                self.log.emit("Training data will not be shuffled")
            
            # Create model based on selected architecture
            self.log.emit(f"Loading model: {self.model_name}")
            
            if self.model_name == "ResNet-18":
                model = models.resnet18(weights='IMAGENET1K_V1')
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif self.model_name == "ResNet-34":
                model = models.resnet34(weights='IMAGENET1K_V1')
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif self.model_name == "ResNet-50":
                model = models.resnet50(weights='IMAGENET1K_V1')
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif self.model_name == "VGG-16":
                model = models.vgg16(weights='IMAGENET1K_V1')
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            elif self.model_name == "VGG-19":
                model = models.vgg19(weights='IMAGENET1K_V1')
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            elif self.model_name == "DenseNet-121":
                model = models.densenet121(weights='IMAGENET1K_V1')
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif self.model_name == "DenseNet-161":
                model = models.densenet161(weights='IMAGENET1K_V1')
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif self.model_name == "MobileNet-V2":
                model = models.mobilenet_v2(weights='IMAGENET1K_V1')
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif self.model_name == "MobileNet-V3 Large":
                model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            elif self.model_name == "EfficientNet-B0":
                model = models.efficientnet_b0(weights='IMAGENET1K_V1')
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif self.model_name == "EfficientNet-B4":
                model = models.efficientnet_b4(weights='IMAGENET1K_V1')
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif self.model_name == "SqueezeNet-1.1":
                model = models.squeezenet1_1(weights='IMAGENET1K_V1')
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            elif self.model_name == "AlexNet":
                model = models.alexnet(weights='IMAGENET1K_V1')
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            elif self.model_name == "Inception-V3":
                model = models.inception_v3(weights='IMAGENET1K_V1')
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                # Default to ResNet-18
                model = models.resnet18(weights='IMAGENET1K_V1')
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            model = model.to(self.device)
            
            # Loss function selection
            if self.loss_function == "CrossEntropyLoss":
                criterion = nn.CrossEntropyLoss()
                self.log.emit("Using CrossEntropyLoss")
            elif self.loss_function == "NLLLoss":
                criterion = nn.NLLLoss()
                self.log.emit("Using NLLLoss (requires log_softmax output)")
            elif self.loss_function == "FocalLoss":
                criterion = FocalLoss(alpha=1, gamma=2)
                self.log.emit("Using FocalLoss (good for imbalanced datasets)")
            elif self.loss_function == "LabelSmoothingLoss":
                criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
                self.log.emit("Using LabelSmoothingLoss with 0.1 smoothing")
            else:
                criterion = nn.CrossEntropyLoss()
                self.log.emit("Using default CrossEntropyLoss")
            
            # Optimizer selection
            if self.optimizer_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            elif self.optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
            elif self.optimizer_name == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
            elif self.optimizer_name == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            else:
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Learning rate scheduler
            scheduler = None
            if self.scheduler_name == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                self.log.emit("Using StepLR: reduces LR by 0.1x every 10 epochs")
            elif self.scheduler_name == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
                self.log.emit("Using ReduceLROnPlateau: reduces LR when val accuracy plateaus")
            elif self.scheduler_name == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
                self.log.emit("Using CosineAnnealingLR: cosine annealing schedule")
            elif self.scheduler_name == "ExponentialLR":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                self.log.emit("Using ExponentialLR: exponential decay with gamma=0.95")
            
            # Early stopping variables
            best_val_acc = 0.0
            epochs_no_improve = 0
            
            # Training loop
            for epoch in range(self.epochs):
                # Reshuffle data if needed
                if self.shuffle_frequency > 0 and epoch > 0 and epoch % self.shuffle_frequency == 0:
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    self.log.emit(f"Reshuffled training data at epoch {epoch+1}")
                
                # Training phase
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                train_accuracy = 100. * correct / total
                train_loss = running_loss / len(train_loader)
                
                # Store training metrics
                self.train_accuracies.append(train_accuracy)
                self.train_losses.append(train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, targets in val_loader:
                        data, targets = data.to(self.device), targets.to(self.device)
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                
                val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0.0
                val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                
                # Store validation metrics
                self.val_accuracies.append(val_accuracy)
                self.val_losses.append(val_loss)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                self.log.emit(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.6f}")
                
                # Update learning rate scheduler
                if scheduler is not None:
                    if self.scheduler_name == "ReduceLROnPlateau":
                        scheduler.step(val_accuracy)
                    else:
                        scheduler.step()
                
                # Save best model and check early stopping
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    best_model_state = model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Early stopping
                if self.patience > 0 and epochs_no_improve >= self.patience:
                    self.log.emit(f"Early stopping triggered after {epoch+1} epochs (no improvement for {self.patience} epochs)")
                    break
                
                # Update progress
                progress_value = int(((epoch + 1) / self.epochs) * 100)
                self.progress.emit(progress_value)

                # Periodic checkpointing and test evaluation (every N epochs)
                try:
                    if hasattr(self, 'save_every_n') and self.save_every_n and self.save_output_dir:
                        if self.save_every_n > 0 and ((epoch + 1) % self.save_every_n) == 0:
                            # Ensure output directory exists
                            os.makedirs(self.save_output_dir, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_fname = os.path.join(self.save_output_dir, f"model_epoch_{epoch+1}_{timestamp}.pth")
                            # Prepare metadata snapshot
                            metadata_snapshot = {
                                'model_architecture': self.model_name,
                                'classes': full_dataset.classes,
                                'num_classes': num_classes,
                                'train_split': self.train_split,
                                'val_split': self.val_split,
                                'test_split': self.test_split,
                                'epoch_saved': epoch + 1,
                                'optimizer': self.optimizer_name,
                                'learning_rate': self.learning_rate,
                                'batch_size': self.batch_size,
                                'loss_function': self.loss_function,
                                'scheduler': self.scheduler_name,
                                'augmentation_settings': self.augmentation_settings
                            }
                            # Save checkpoint
                            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'metadata': metadata_snapshot}, model_fname)

                            # Run test evaluation on current model state
                            model.eval()
                            t_preds = []
                            t_labels = []
                            t_probs = []
                            with torch.no_grad():
                                for data, targets in test_loader:
                                    data, targets = data.to(self.device), targets.to(self.device)
                                    outputs = model(data)
                                    probs = torch.softmax(outputs, dim=1)
                                    _, preds = outputs.max(1)
                                    t_preds.extend(preds.cpu().numpy())
                                    t_labels.extend(targets.cpu().numpy())
                                    t_probs.extend(probs.cpu().numpy())

                            # Compute metrics
                            if len(t_labels) > 0:
                                test_acc = 100. * (np.array(t_preds) == np.array(t_labels)).sum() / len(t_labels)
                                cm = confusion_matrix(t_labels, t_preds).tolist()
                                precision = precision_score(t_labels, t_preds, average='macro', zero_division=0)
                                recall = recall_score(t_labels, t_preds, average='macro', zero_division=0)
                                f1 = f1_score(t_labels, t_preds, average='macro', zero_division=0)
                            else:
                                test_acc = 0.0
                                cm = []
                                precision = recall = f1 = 0.0

                            metrics = {
                                'epoch': epoch + 1,
                                'timestamp': timestamp,
                                'test_accuracy': test_acc,
                                'precision_macro': precision,
                                'recall_macro': recall,
                                'f1_macro': f1,
                                'confusion_matrix': cm,
                                'num_test_samples': len(t_labels)
                            }

                            metrics_fname = os.path.join(self.save_output_dir, f"metrics_epoch_{epoch+1}_{timestamp}.json")
                            with open(metrics_fname, 'w') as mf:
                                json.dump(metrics, mf, indent=4)

                            preds_fname = os.path.join(self.save_output_dir, f"preds_epoch_{epoch+1}_{timestamp}.npz")
                            np.savez(preds_fname, predictions=np.array(t_preds), labels=np.array(t_labels), probabilities=np.array(t_probs))

                            self.log.emit(f"Periodic save: checkpoint and metrics saved for epoch {epoch+1} -> {self.save_output_dir}")
                except Exception as e:
                    self.log.emit(f"Error during periodic save at epoch {epoch+1}: {str(e)}")
            
            # Test phase on best model
            if best_val_acc > 0:
                model.load_state_dict(best_model_state)
            
            model.eval()
            test_correct = 0
            test_total = 0
            
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
                    
                    # Store predictions and labels
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
            
            test_accuracy = 100. * test_correct / test_total if test_total > 0 else 0.0
            self.log.emit(f"\nTest Set Performance - Accuracy: {test_accuracy:.2f}%")
            
            # Store predictions for statistics
            self.test_predictions = np.array(all_predictions)
            self.test_labels = np.array(all_labels)
            self.test_probabilities = np.array(all_probabilities)
            
            # Store model and metadata for export
            self.trained_model = model
            self.model_metadata = {
                'model_architecture': self.model_name,
                'classes': full_dataset.classes,
                'num_classes': num_classes,
                'train_split': self.train_split,
                'val_split': self.val_split,
                'test_split': self.test_split,
                'best_val_accuracy': best_val_acc,
                'test_accuracy': test_accuracy,
                'optimizer': self.optimizer_name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs_trained': epoch + 1 if 'epoch' in locals() else self.epochs,
                'loss_function': self.loss_function,
                'scheduler': self.scheduler_name,
                'augmentation': self.augmentation_settings
            }
            
            self.log.emit(f"\nTraining completed! Use 'Export Model' button to save the trained model.")
            self.finished.emit()
            
        except Exception as e:
            self.log.emit(f"Error during training: {str(e)}")

class ChipTrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_path = ""
        self.training_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("ChipTrainer - Neural Network Image Classifier")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("ChipTrainer - Neural Network Image Classifier")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Data selection group
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        self.dir_button = QPushButton("Select Data Directory")
        self.dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(QLabel("Data Directory:"))
        dir_layout.addWidget(self.dir_label, 1)
        dir_layout.addWidget(self.dir_button)
        data_layout.addLayout(dir_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Architecture:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "ResNet-18",
            "ResNet-34",
            "ResNet-50",
            "VGG-16",
            "VGG-19",
            "DenseNet-121",
            "DenseNet-161",
            "MobileNet-V2",
            "MobileNet-V3 Large",
            "EfficientNet-B0",
            "EfficientNet-B4",
            "SqueezeNet-1.1",
            "AlexNet",
            "Inception-V3"
        ])
        self.model_combo.setCurrentText("ResNet-18")
        model_layout.addWidget(self.model_combo, 1)
        data_layout.addLayout(model_layout)
        
        # Info label
        info_label = QLabel("Select a directory containing subfolders with images.\nEach subfolder name will be used as a class label.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(info_label)
        
        layout.addWidget(data_group)
        
        # Data split group
        split_group = QGroupBox("Data Split Proportions")
        split_layout = QGridLayout(split_group)
        
        # Training split
        split_layout.addWidget(QLabel("Training:"), 0, 0)
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.0, 1.0)
        self.train_split_spin.setDecimals(2)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.70)
        self.train_split_spin.valueChanged.connect(self.update_split_validation)
        split_layout.addWidget(self.train_split_spin, 0, 1)
        split_layout.addWidget(QLabel("(70%)"), 0, 2)
        
        # Validation split
        split_layout.addWidget(QLabel("Validation:"), 1, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 1.0)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.15)
        self.val_split_spin.valueChanged.connect(self.update_split_validation)
        split_layout.addWidget(self.val_split_spin, 1, 1)
        split_layout.addWidget(QLabel("(15%)"), 1, 2)
        
        # Test split
        split_layout.addWidget(QLabel("Test:"), 2, 0)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.0, 1.0)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.15)
        self.test_split_spin.valueChanged.connect(self.update_split_validation)
        split_layout.addWidget(self.test_split_spin, 2, 1)
        split_layout.addWidget(QLabel("(15%)"), 2, 2)
        
        # Total label
        self.split_total_label = QLabel("Total: 1.00")
        self.split_total_label.setStyleSheet("font-weight: bold;")
        split_layout.addWidget(self.split_total_label, 3, 0, 1, 3)
        
        layout.addWidget(split_group)
        
        # Training parameters group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)
        
        # Epochs
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate:"), 0, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        params_layout.addWidget(self.lr_spin, 0, 3)
        
        # Batch size
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        params_layout.addWidget(self.batch_spin, 1, 1)
        
        # Optimizer
        params_layout.addWidget(QLabel("Optimizer:"), 1, 2)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        self.optimizer_combo.setCurrentText("Adam")
        params_layout.addWidget(self.optimizer_combo, 1, 3)
        
        # Learning rate scheduler
        params_layout.addWidget(QLabel("LR Scheduler:"), 2, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems([
            "None",
            "StepLR",
            "ReduceLROnPlateau",
            "CosineAnnealingLR",
            "ExponentialLR"
        ])
        self.scheduler_combo.setCurrentText("ReduceLROnPlateau")
        params_layout.addWidget(self.scheduler_combo, 2, 1)
        
        # Early stopping patience
        params_layout.addWidget(QLabel("Early Stop Patience:"), 2, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(5)
        self.patience_spin.setSpecialValueText("Disabled")
        params_layout.addWidget(self.patience_spin, 2, 3)
        
        # Data augmentation
        params_layout.addWidget(QLabel("Data Augmentation:"), 3, 0)
        self.augmentation_button = QPushButton("Configure Augmentations")
        self.augmentation_button.clicked.connect(self.open_augmentation_dialog)
        params_layout.addWidget(self.augmentation_button, 3, 1)
        
        # Store augmentation settings
        self.augmentation_settings = {}
        # Periodic save output directory (default empty)
        self.save_output_dir = ""
        
        # Loss function
        params_layout.addWidget(QLabel("Loss Function:"), 3, 2)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems([
            "CrossEntropyLoss",
            "NLLLoss",
            "FocalLoss",
            "LabelSmoothingLoss"
        ])
        self.loss_combo.setCurrentText("CrossEntropyLoss")
        params_layout.addWidget(self.loss_combo, 3, 3)
        
        # Shuffle frequency
        params_layout.addWidget(QLabel("Shuffle Every N Epochs:"), 4, 0)
        self.shuffle_spin = QSpinBox()
        self.shuffle_spin.setRange(0, 100)
        self.shuffle_spin.setValue(1)
        self.shuffle_spin.setSpecialValueText("Never")
        self.shuffle_spin.setToolTip("How often to reshuffle the training data (0 = never, 1 = every epoch)")
        params_layout.addWidget(self.shuffle_spin, 4, 1)

        # Save every N epochs control
        params_layout.addWidget(QLabel("Save Every N Epochs:"), 5, 0)
        self.save_every_spin = QSpinBox()
        self.save_every_spin.setRange(0, 1000)
        self.save_every_spin.setValue(0)
        self.save_every_spin.setSpecialValueText("Disabled")
        self.save_every_spin.setToolTip("Save model and test metrics every N epochs (0 = disabled)")
        params_layout.addWidget(self.save_every_spin, 5, 1)

        # Save directory selector
        params_layout.addWidget(QLabel("Save Directory:"), 6, 0)
        save_dir_layout = QHBoxLayout()
        self.save_dir_label = QLabel("No save directory selected")
        self.save_dir_button = QPushButton("Select Save Directory")
        self.save_dir_button.clicked.connect(self.select_save_directory)
        save_dir_layout.addWidget(self.save_dir_label)
        save_dir_layout.addWidget(self.save_dir_button)
        params_layout.addLayout(save_dir_layout, 6, 1, 1, 3)
        
        layout.addWidget(params_group)
        
        # Config save/load controls
        config_layout = QHBoxLayout()
        self.save_config_button = QPushButton("Save Configuration")
        self.save_config_button.clicked.connect(self.save_config)
        self.load_config_button = QPushButton("Load Configuration")
        self.load_config_button.clicked.connect(self.load_config)
        
        config_layout.addWidget(self.save_config_button)
        config_layout.addWidget(self.load_config_button)
        config_layout.addStretch()
        
        layout.addLayout(config_layout)
        
        # Training controls
        controls_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        
        self.export_button = QPushButton("Export Model")
        self.export_button.clicked.connect(self.export_model)
        self.export_button.setEnabled(False)
        
        self.export_stats_button = QPushButton("Export Stats")
        self.export_stats_button.clicked.connect(self.export_stats)
        self.export_stats_button.setEnabled(False)
        
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.load_model_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.export_stats_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)
        
        # Device info - show PyTorch availability
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.device_info = QLabel(f"PyTorch Device: CUDA - {device_name} (CUDA {cuda_version})")
                self.device_info.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.device_info = QLabel("PyTorch Device: CPU")
                self.device_info.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.device_info = QLabel(f"PyTorch: Not Available - {TORCH_ERROR}")
            self.device_info.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.device_info)
    
    def update_split_validation(self):
        """Validate that splits sum to 1.0 and update labels"""
        train_val = self.train_split_spin.value()
        val_val = self.val_split_spin.value()
        test_val = self.test_split_spin.value()
        
        total = train_val + val_val + test_val
        
        # Update total label
        self.split_total_label.setText(f"Total: {total:.2f}")
        
        # Color code based on validity
        if abs(total - 1.0) < 0.01:  # Allow small rounding errors
            self.split_total_label.setStyleSheet("font-weight: bold; color: green;")
            self.train_button.setEnabled(bool(self.data_path))
        else:
            self.split_total_label.setStyleSheet("font-weight: bold; color: red;")
            self.train_button.setEnabled(False)
        
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.data_path = directory
            self.dir_label.setText(directory)
            
            # Check for subfolders
            try:
                subfolders = [d for d in os.listdir(directory) 
                             if os.path.isdir(os.path.join(directory, d))]
                
                if subfolders:
                    self.log_text.append(f"Found {len(subfolders)} classes: {', '.join(subfolders)}")
                    self.train_button.setEnabled(True)
                else:
                    self.log_text.append("Warning: No subfolders found in selected directory!")
                    self.train_button.setEnabled(False)
            except Exception as e:
                self.log_text.append(f"Error reading directory: {str(e)}")
                self.train_button.setEnabled(False)
    
    def open_augmentation_dialog(self):
        """Open dialog to configure data augmentation options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Data Augmentation")
        dialog.setMinimumWidth(700)
        dialog.setMinimumHeight(600)
        
        main_layout = QVBoxLayout()
        
        # Create scroll area for augmentation options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Dictionary to store augmentation widgets
        aug_widgets = {}
        
        # Geometric transformations group
        geo_group = QGroupBox("Geometric Transformations")
        geo_layout = QFormLayout()
        
        # RandomHorizontalFlip
        hflip_check = QCheckBox("Random Horizontal Flip")
        hflip_prob = QDoubleSpinBox()
        hflip_prob.setRange(0.0, 1.0)
        hflip_prob.setSingleStep(0.1)
        hflip_prob.setValue(0.5)
        hflip_prob.setPrefix("p=")
        hflip_layout = QHBoxLayout()
        hflip_layout.addWidget(hflip_check)
        hflip_layout.addWidget(hflip_prob)
        hflip_layout.addStretch()
        geo_layout.addRow(hflip_layout)
        aug_widgets['RandomHorizontalFlip'] = (hflip_check, hflip_prob)
        
        # RandomVerticalFlip
        vflip_check = QCheckBox("Random Vertical Flip")
        vflip_prob = QDoubleSpinBox()
        vflip_prob.setRange(0.0, 1.0)
        vflip_prob.setSingleStep(0.1)
        vflip_prob.setValue(0.5)
        vflip_prob.setPrefix("p=")
        vflip_layout = QHBoxLayout()
        vflip_layout.addWidget(vflip_check)
        vflip_layout.addWidget(vflip_prob)
        vflip_layout.addStretch()
        geo_layout.addRow(vflip_layout)
        aug_widgets['RandomVerticalFlip'] = (vflip_check, vflip_prob)
        
        # RandomRotation
        rotate_check = QCheckBox("Random Rotation")
        rotate_degrees = QSpinBox()
        rotate_degrees.setRange(0, 180)
        rotate_degrees.setValue(15)
        rotate_degrees.setSuffix("°")
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(rotate_check)
        rotate_layout.addWidget(QLabel("degrees:"))
        rotate_layout.addWidget(rotate_degrees)
        rotate_layout.addStretch()
        geo_layout.addRow(rotate_layout)
        aug_widgets['RandomRotation'] = (rotate_check, rotate_degrees)
        
        # RandomCrop
        crop_check = QCheckBox("Random Crop")
        crop_size = QSpinBox()
        crop_size.setRange(32, 512)
        crop_size.setValue(224)
        crop_size.setSuffix("px")
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(crop_check)
        crop_layout.addWidget(QLabel("size:"))
        crop_layout.addWidget(crop_size)
        crop_layout.addStretch()
        geo_layout.addRow(crop_layout)
        aug_widgets['RandomCrop'] = (crop_check, crop_size)
        
        # RandomPerspective
        persp_check = QCheckBox("Random Perspective")
        persp_prob = QDoubleSpinBox()
        persp_prob.setRange(0.0, 1.0)
        persp_prob.setSingleStep(0.1)
        persp_prob.setValue(0.5)
        persp_prob.setPrefix("p=")
        persp_layout = QHBoxLayout()
        persp_layout.addWidget(persp_check)
        persp_layout.addWidget(persp_prob)
        persp_layout.addStretch()
        geo_layout.addRow(persp_layout)
        aug_widgets['RandomPerspective'] = (persp_check, persp_prob)
        
        geo_group.setLayout(geo_layout)
        scroll_layout.addWidget(geo_group)
        
        # Color transformations group
        color_group = QGroupBox("Color Transformations")
        color_layout = QFormLayout()
        
        # ColorJitter
        jitter_check = QCheckBox("Color Jitter")
        jitter_brightness = QDoubleSpinBox()
        jitter_brightness.setRange(0.0, 1.0)
        jitter_brightness.setSingleStep(0.1)
        jitter_brightness.setValue(0.2)
        jitter_contrast = QDoubleSpinBox()
        jitter_contrast.setRange(0.0, 1.0)
        jitter_contrast.setSingleStep(0.1)
        jitter_contrast.setValue(0.2)
        jitter_saturation = QDoubleSpinBox()
        jitter_saturation.setRange(0.0, 1.0)
        jitter_saturation.setSingleStep(0.1)
        jitter_saturation.setValue(0.2)
        jitter_hue = QDoubleSpinBox()
        jitter_hue.setRange(0.0, 0.5)
        jitter_hue.setSingleStep(0.05)
        jitter_hue.setValue(0.1)
        jitter_layout = QHBoxLayout()
        jitter_layout.addWidget(jitter_check)
        jitter_layout.addWidget(QLabel("B:"))
        jitter_layout.addWidget(jitter_brightness)
        jitter_layout.addWidget(QLabel("C:"))
        jitter_layout.addWidget(jitter_contrast)
        jitter_layout.addWidget(QLabel("S:"))
        jitter_layout.addWidget(jitter_saturation)
        jitter_layout.addWidget(QLabel("H:"))
        jitter_layout.addWidget(jitter_hue)
        jitter_layout.addStretch()
        color_layout.addRow(jitter_layout)
        aug_widgets['ColorJitter'] = (jitter_check, jitter_brightness, jitter_contrast, jitter_saturation, jitter_hue)
        
        # RandomGrayscale
        gray_check = QCheckBox("Random Grayscale")
        gray_prob = QDoubleSpinBox()
        gray_prob.setRange(0.0, 1.0)
        gray_prob.setSingleStep(0.1)
        gray_prob.setValue(0.1)
        gray_prob.setPrefix("p=")
        gray_layout = QHBoxLayout()
        gray_layout.addWidget(gray_check)
        gray_layout.addWidget(gray_prob)
        gray_layout.addStretch()
        color_layout.addRow(gray_layout)
        aug_widgets['RandomGrayscale'] = (gray_check, gray_prob)
        
        # GaussianBlur
        blur_check = QCheckBox("Gaussian Blur")
        blur_kernel = QSpinBox()
        blur_kernel.setRange(3, 23)
        blur_kernel.setSingleStep(2)
        blur_kernel.setValue(5)
        blur_sigma_min = QDoubleSpinBox()
        blur_sigma_min.setRange(0.1, 2.0)
        blur_sigma_min.setSingleStep(0.1)
        blur_sigma_min.setValue(0.1)
        blur_sigma_max = QDoubleSpinBox()
        blur_sigma_max.setRange(0.1, 2.0)
        blur_sigma_max.setSingleStep(0.1)
        blur_sigma_max.setValue(2.0)
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(blur_check)
        blur_layout.addWidget(QLabel("kernel:"))
        blur_layout.addWidget(blur_kernel)
        blur_layout.addWidget(QLabel("σ:"))
        blur_layout.addWidget(blur_sigma_min)
        blur_layout.addWidget(QLabel("-"))
        blur_layout.addWidget(blur_sigma_max)
        blur_layout.addStretch()
        color_layout.addRow(blur_layout)
        aug_widgets['GaussianBlur'] = (blur_check, blur_kernel, blur_sigma_min, blur_sigma_max)
        
        # RandomAdjustSharpness
        sharp_check = QCheckBox("Random Adjust Sharpness")
        sharp_factor = QDoubleSpinBox()
        sharp_factor.setRange(0.0, 4.0)
        sharp_factor.setSingleStep(0.5)
        sharp_factor.setValue(2.0)
        sharp_prob = QDoubleSpinBox()
        sharp_prob.setRange(0.0, 1.0)
        sharp_prob.setSingleStep(0.1)
        sharp_prob.setValue(0.5)
        sharp_prob.setPrefix("p=")
        sharp_layout = QHBoxLayout()
        sharp_layout.addWidget(sharp_check)
        sharp_layout.addWidget(QLabel("factor:"))
        sharp_layout.addWidget(sharp_factor)
        sharp_layout.addWidget(sharp_prob)
        sharp_layout.addStretch()
        color_layout.addRow(sharp_layout)
        aug_widgets['RandomAdjustSharpness'] = (sharp_check, sharp_factor, sharp_prob)
        
        color_group.setLayout(color_layout)
        scroll_layout.addWidget(color_group)
        
        # Advanced augmentation group
        advanced_group = QGroupBox("Advanced Augmentations")
        advanced_layout = QFormLayout()
        
        # RandomErasing
        erase_check = QCheckBox("Random Erasing")
        erase_prob = QDoubleSpinBox()
        erase_prob.setRange(0.0, 1.0)
        erase_prob.setSingleStep(0.1)
        erase_prob.setValue(0.5)
        erase_prob.setPrefix("p=")
        erase_scale_min = QDoubleSpinBox()
        erase_scale_min.setRange(0.01, 0.5)
        erase_scale_min.setSingleStep(0.01)
        erase_scale_min.setValue(0.02)
        erase_scale_max = QDoubleSpinBox()
        erase_scale_max.setRange(0.01, 0.5)
        erase_scale_max.setSingleStep(0.01)
        erase_scale_max.setValue(0.33)
        erase_layout = QHBoxLayout()
        erase_layout.addWidget(erase_check)
        erase_layout.addWidget(erase_prob)
        erase_layout.addWidget(QLabel("scale:"))
        erase_layout.addWidget(erase_scale_min)
        erase_layout.addWidget(QLabel("-"))
        erase_layout.addWidget(erase_scale_max)
        erase_layout.addStretch()
        advanced_layout.addRow(erase_layout)
        aug_widgets['RandomErasing'] = (erase_check, erase_prob, erase_scale_min, erase_scale_max)
        
        advanced_group.setLayout(advanced_layout)
        scroll_layout.addWidget(advanced_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # Load existing settings
        for aug_name, widgets in aug_widgets.items():
            if aug_name in self.augmentation_settings:
                settings = self.augmentation_settings[aug_name]
                if 'enabled' in settings:
                    widgets[0].setChecked(settings['enabled'])
                if aug_name == 'RandomHorizontalFlip' and 'p' in settings:
                    widgets[1].setValue(settings['p'])
                elif aug_name == 'RandomVerticalFlip' and 'p' in settings:
                    widgets[1].setValue(settings['p'])
                elif aug_name == 'RandomRotation' and 'degrees' in settings:
                    widgets[1].setValue(settings['degrees'])
                elif aug_name == 'RandomCrop' and 'size' in settings:
                    widgets[1].setValue(settings['size'])
                elif aug_name == 'RandomPerspective' and 'p' in settings:
                    widgets[1].setValue(settings['p'])
                elif aug_name == 'ColorJitter':
                    if 'brightness' in settings:
                        widgets[1].setValue(settings['brightness'])
                    if 'contrast' in settings:
                        widgets[2].setValue(settings['contrast'])
                    if 'saturation' in settings:
                        widgets[3].setValue(settings['saturation'])
                    if 'hue' in settings:
                        widgets[4].setValue(settings['hue'])
                elif aug_name == 'RandomGrayscale' and 'p' in settings:
                    widgets[1].setValue(settings['p'])
                elif aug_name == 'GaussianBlur':
                    if 'kernel_size' in settings:
                        widgets[1].setValue(settings['kernel_size'])
                    if 'sigma_min' in settings:
                        widgets[2].setValue(settings['sigma_min'])
                    if 'sigma_max' in settings:
                        widgets[3].setValue(settings['sigma_max'])
                elif aug_name == 'RandomAdjustSharpness':
                    if 'sharpness_factor' in settings:
                        widgets[1].setValue(settings['sharpness_factor'])
                    if 'p' in settings:
                        widgets[2].setValue(settings['p'])
                elif aug_name == 'RandomErasing':
                    if 'p' in settings:
                        widgets[1].setValue(settings['p'])
                    if 'scale_min' in settings:
                        widgets[2].setValue(settings['scale_min'])
                    if 'scale_max' in settings:
                        widgets[3].setValue(settings['scale_max'])
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        def save_settings():
            self.augmentation_settings = {}
            
            # Save RandomHorizontalFlip
            check, prob = aug_widgets['RandomHorizontalFlip']
            if check.isChecked():
                self.augmentation_settings['RandomHorizontalFlip'] = {'enabled': True, 'p': prob.value()}
            
            # Save RandomVerticalFlip
            check, prob = aug_widgets['RandomVerticalFlip']
            if check.isChecked():
                self.augmentation_settings['RandomVerticalFlip'] = {'enabled': True, 'p': prob.value()}
            
            # Save RandomRotation
            check, degrees = aug_widgets['RandomRotation']
            if check.isChecked():
                self.augmentation_settings['RandomRotation'] = {'enabled': True, 'degrees': degrees.value()}
            
            # Save RandomCrop
            check, size = aug_widgets['RandomCrop']
            if check.isChecked():
                self.augmentation_settings['RandomCrop'] = {'enabled': True, 'size': size.value()}
            
            # Save RandomPerspective
            check, prob = aug_widgets['RandomPerspective']
            if check.isChecked():
                self.augmentation_settings['RandomPerspective'] = {'enabled': True, 'p': prob.value()}
            
            # Save ColorJitter
            check, brightness, contrast, saturation, hue = aug_widgets['ColorJitter']
            if check.isChecked():
                self.augmentation_settings['ColorJitter'] = {
                    'enabled': True,
                    'brightness': brightness.value(),
                    'contrast': contrast.value(),
                    'saturation': saturation.value(),
                    'hue': hue.value()
                }
            
            # Save RandomGrayscale
            check, prob = aug_widgets['RandomGrayscale']
            if check.isChecked():
                self.augmentation_settings['RandomGrayscale'] = {'enabled': True, 'p': prob.value()}
            
            # Save GaussianBlur
            check, kernel, sigma_min, sigma_max = aug_widgets['GaussianBlur']
            if check.isChecked():
                self.augmentation_settings['GaussianBlur'] = {
                    'enabled': True,
                    'kernel_size': kernel.value(),
                    'sigma_min': sigma_min.value(),
                    'sigma_max': sigma_max.value()
                }
            
            # Save RandomAdjustSharpness
            check, factor, prob = aug_widgets['RandomAdjustSharpness']
            if check.isChecked():
                self.augmentation_settings['RandomAdjustSharpness'] = {
                    'enabled': True,
                    'sharpness_factor': factor.value(),
                    'p': prob.value()
                }
            
            # Save RandomErasing
            check, prob, scale_min, scale_max = aug_widgets['RandomErasing']
            if check.isChecked():
                self.augmentation_settings['RandomErasing'] = {
                    'enabled': True,
                    'p': prob.value(),
                    'scale_min': scale_min.value(),
                    'scale_max': scale_max.value()
                }
            
            # Update button text
            num_enabled = len(self.augmentation_settings)
            if num_enabled == 0:
                self.augmentation_button.setText("Configure Augmentations (None)")
            else:
                self.augmentation_button.setText(f"Configure Augmentations ({num_enabled} enabled)")
            
            dialog.accept()
        
        ok_button.clicked.connect(save_settings)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)
        
        dialog.setLayout(main_layout)
        dialog.exec_()
    
    def start_training(self):
        if not self.data_path:
            return
        
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            QMessageBox.critical(self, "Error", f"PyTorch is not available.\n\nError: {TORCH_ERROR}\n\nPlease ensure PyTorch is installed correctly.")
            return
            
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # Create and start training thread
        self.training_thread = TrainingThread(
            self.data_path,
            self.epochs_spin.value(),
            self.lr_spin.value(),
            self.batch_spin.value(),
            self.train_split_spin.value(),
            self.val_split_spin.value(),
            self.test_split_spin.value(),
            self.model_combo.currentText(),
            self.optimizer_combo.currentText(),
            self.scheduler_combo.currentText(),
            self.patience_spin.value(),
            self.augmentation_settings,
            self.loss_combo.currentText(),
            self.shuffle_spin.value()
        )

        # Pass periodic save settings to the training thread
        try:
            self.training_thread.save_every_n = int(self.save_every_spin.value())
            self.training_thread.save_output_dir = self.save_output_dir if getattr(self, 'save_output_dir', None) else ""
        except Exception:
            # Ignore if not set; training thread has defaults
            pass
        
        self.training_thread.progress.connect(self.progress_bar.setValue)
        self.training_thread.log.connect(self.log_text.append)
        self.training_thread.log.connect(self.update_device_info)
        self.training_thread.finished.connect(self.training_finished)
        
        self.training_thread.start()

    def select_save_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.save_output_dir = directory
            self.save_dir_label.setText(directory)
            self.log_text.append(f"Save directory set to: {directory}")
    
    def update_device_info(self, message):
        """Update device info when training starts"""
        if "Starting training on device:" in message:
            device = message.split(":")[-1].strip()
            self.device_info.setText(f"PyTorch Device: {device}")
            if "cuda" in device.lower():
                self.device_info.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.device_info.setStyleSheet("color: blue; font-weight: bold;")
    
    def save_config(self):
        """Save current configuration to a JSON file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Configuration", 
            "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            config = {
                "data_path": self.data_path,
                "model_architecture": self.model_combo.currentText(),
                "epochs": self.epochs_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "batch_size": self.batch_spin.value(),
                "optimizer": self.optimizer_combo.currentText(),
                "lr_scheduler": self.scheduler_combo.currentText(),
                "early_stop_patience": self.patience_spin.value(),
                "augmentation_settings": self.augmentation_settings,
                "loss_function": self.loss_combo.currentText(),
                "shuffle_frequency": self.shuffle_spin.value(),
                "train_split": self.train_split_spin.value(),
                "val_split": self.val_split_spin.value(),
                "test_split": self.test_split_spin.value()
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
                self.log_text.append(f"Configuration saved to: {filename}")
                QMessageBox.information(self, "Success", f"Configuration saved successfully to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")
    
    def load_config(self):
        """Load configuration from a JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Configuration", 
            "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Load data path
                if "data_path" in config and config["data_path"]:
                    self.data_path = config["data_path"]
                    self.dir_label.setText(config["data_path"])
                    # Check if directory still exists
                    if os.path.exists(config["data_path"]):
                        subfolders = [d for d in os.listdir(config["data_path"]) 
                                     if os.path.isdir(os.path.join(config["data_path"], d))]
                        if subfolders:
                            self.log_text.append(f"Found {len(subfolders)} classes: {', '.join(subfolders)}")
                    else:
                        self.log_text.append("Warning: Saved data path no longer exists!")
                
                # Load model architecture
                if "model_architecture" in config:
                    self.model_combo.setCurrentText(config["model_architecture"])
                
                # Load training parameters
                if "epochs" in config:
                    self.epochs_spin.setValue(config["epochs"])
                if "learning_rate" in config:
                    self.lr_spin.setValue(config["learning_rate"])
                if "batch_size" in config:
                    self.batch_spin.setValue(config["batch_size"])
                if "optimizer" in config:
                    self.optimizer_combo.setCurrentText(config["optimizer"])
                if "lr_scheduler" in config:
                    self.scheduler_combo.setCurrentText(config["lr_scheduler"])
                if "early_stop_patience" in config:
                    self.patience_spin.setValue(config["early_stop_patience"])
                if "augmentation_settings" in config:
                    self.augmentation_settings = config["augmentation_settings"]
                    num_enabled = len(self.augmentation_settings)
                    if num_enabled == 0:
                        self.augmentation_button.setText("Configure Augmentations (None)")
                    else:
                        self.augmentation_button.setText(f"Configure Augmentations ({num_enabled} enabled)")
                if "loss_function" in config:
                    self.loss_combo.setCurrentText(config["loss_function"])
                if "shuffle_frequency" in config:
                    self.shuffle_spin.setValue(config["shuffle_frequency"])
                
                # Load data splits
                if "train_split" in config:
                    self.train_split_spin.setValue(config["train_split"])
                if "val_split" in config:
                    self.val_split_spin.setValue(config["val_split"])
                if "test_split" in config:
                    self.test_split_spin.setValue(config["test_split"])
                
                # Update validation
                self.update_split_validation()
                
                self.log_text.append(f"Configuration loaded from: {filename}")
                QMessageBox.information(self, "Success", f"Configuration loaded successfully from:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")
    
    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            self.training_finished()
            self.log_text.append("Training stopped by user.")
    
    def training_finished(self):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.export_stats_button.setEnabled(True)
        self.progress_bar.setValue(100)
    
    def load_model(self):
        """Load a previously exported model"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "PyTorch Models (*.pth *.pt);;All Files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            # Load the model file
            self.log_text.append(f"Loading model from: {filename}")
            
            # Check if it's a TorchScript model
            if filename.endswith('.pt'):
                # TorchScript models need to be loaded differently
                try:
                    model = torch.jit.load(filename, map_location='cpu')
                    self.log_text.append("Loaded TorchScript model")
                    
                    # Try to load metadata from companion JSON file
                    metadata_file = filename.replace('.pt', '_metadata.json')
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        model_arch = metadata.get('model_architecture', 'Unknown')
                        num_classes = metadata.get('num_classes', 0)
                        classes = metadata.get('classes', [])
                        
                        self.log_text.append(f"Model architecture: {model_arch}")
                        self.log_text.append(f"Model trained on {num_classes} classes: {', '.join(classes)}")
                        
                        if 'best_val_accuracy' in metadata:
                            self.log_text.append(f"Best validation accuracy: {metadata['best_val_accuracy']:.2f}%")
                        if 'test_accuracy' in metadata:
                            self.log_text.append(f"Test accuracy: {metadata['test_accuracy']:.2f}%")
                    else:
                        self.log_text.append("Warning: No metadata file found")
                        metadata = {}
                        model_arch = 'Unknown'
                        num_classes = 0
                        classes = []
                    
                    # Create a pseudo training thread to store the model
                    if not self.training_thread:
                        self.training_thread = TrainingThread(
                            data_path="",
                            model_name=model_arch,
                            epochs=1,
                            learning_rate=0.001,
                            batch_size=32,
                            train_split=0.7,
                            val_split=0.15,
                            test_split=0.15,
                            optimizer_name="Adam",
                            scheduler_name="None",
                            patience=10,
                            augmentation_settings={},
                            loss_function="CrossEntropyLoss",
                            shuffle_frequency=1
                        )
                    
                    self.training_thread.trained_model = model
                    self.training_thread.model_metadata = metadata if metadata else {
                        'model_architecture': model_arch,
                        'classes': classes,
                        'num_classes': num_classes
                    }
                    
                    self.export_button.setEnabled(True)
                    self.log_text.append("TorchScript model loaded successfully! You can export it to different formats.")
                    QMessageBox.information(
                        self,
                        "Success",
                        f"TorchScript model loaded successfully!\n\nArchitecture: {model_arch}\nClasses: {num_classes}"
                    )
                    return
                    
                except Exception as ts_error:
                    self.log_text.append(f"Not a TorchScript model, trying as regular checkpoint: {str(ts_error)}")
            
            # Try loading as regular PyTorch checkpoint
            # Use weights_only=False for compatibility with older models
            checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with metadata
                model_arch = checkpoint.get('model_architecture', 'resnet18')
                num_classes = checkpoint.get('num_classes', 10)
                
                # Load metadata if available
                if 'classes' in checkpoint:
                    classes = checkpoint['classes']
                    self.log_text.append(f"Model trained on {num_classes} classes: {', '.join(classes)}")
                
                if 'best_val_accuracy' in checkpoint:
                    self.log_text.append(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.2f}%")
                
                if 'test_accuracy' in checkpoint:
                    self.log_text.append(f"Test accuracy: {checkpoint['test_accuracy']:.2f}%")
                
                # Create model architecture
                self.log_text.append(f"Loading {model_arch} architecture...")
                
                # Normalize architecture name (handle different naming conventions)
                model_arch_normalized = model_arch.lower().replace('-', '').replace('_', '').replace(' ', '')
                
                # Map of normalized names to actual model functions
                model_mapping = {
                    'resnet18': models.resnet18,
                    'resnet34': models.resnet34,
                    'resnet50': models.resnet50,
                    'vgg16': models.vgg16,
                    'vgg19': models.vgg19,
                    'densenet121': models.densenet121,
                    'densenet169': models.densenet169,
                    'mobilenetv2': models.mobilenet_v2,
                    'mobilenetv3small': models.mobilenet_v3_small,
                    'mobilenetv3large': models.mobilenet_v3_large,
                    'efficientnetb0': models.efficientnet_b0,
                    'efficientnetb1': models.efficientnet_b1,
                    'shufflenetv2x10': models.shufflenet_v2_x1_0,
                    'squeezenet11': models.squeezenet1_1
                }
                
                # Get the model creation function
                if model_arch_normalized in model_mapping:
                    model = model_mapping[model_arch_normalized](weights=None)
                else:
                    raise ValueError(f"Unknown model architecture: {model_arch}")
                
                # Modify final layer to match num_classes
                if hasattr(model, 'fc'):
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                elif hasattr(model, 'classifier'):
                    if isinstance(model.classifier, nn.Sequential):
                        if hasattr(model.classifier[-1], 'in_features'):
                            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                    elif hasattr(model.classifier, 'in_features'):
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                
                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                state_dict = checkpoint['model_state_dict']
                
            elif isinstance(checkpoint, dict):
                # Just a state dict
                QMessageBox.warning(
                    self,
                    "Incomplete Model",
                    "This file contains only model weights without metadata.\n"
                    "To load it, you need to manually select the correct architecture and number of classes.\n"
                    "Please use the full .pth export format for complete model loading."
                )
                return
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Format",
                    "Unable to load this model file. Please use a .pth file exported from this application."
                )
                return
            
            # Create a pseudo training thread to store the model
            if not self.training_thread:
                self.training_thread = TrainingThread(
                    data_path="",
                    model_name=model_arch,
                    epochs=1,
                    learning_rate=0.001,
                    batch_size=32,
                    train_split=0.7,
                    val_split=0.15,
                    test_split=0.15,
                    optimizer_name="Adam",
                    scheduler_name="None",
                    patience=10,
                    augmentation_settings={},
                    loss_function="CrossEntropyLoss",
                    shuffle_frequency=1
                )
            
            self.training_thread.trained_model = model
            self.training_thread.model_metadata = {
                'model_architecture': model_arch,
                'classes': checkpoint.get('classes', []),
                'num_classes': num_classes,
                'train_split': checkpoint.get('train_split', 0.7),
                'val_split': checkpoint.get('val_split', 0.15),
                'test_split': checkpoint.get('test_split', 0.15),
                'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
                'test_accuracy': checkpoint.get('test_accuracy', 0.0),
                'optimizer': checkpoint.get('optimizer', 'Unknown'),
                'learning_rate': checkpoint.get('learning_rate', 0.001),
                'batch_size': checkpoint.get('batch_size', 32),
                'epochs_trained': checkpoint.get('epochs_trained', 0),
                'loss_function': checkpoint.get('loss_function', 'CrossEntropyLoss'),
                'scheduler': checkpoint.get('scheduler', 'None'),
                'augmentation': checkpoint.get('augmentation', 'None')
            }
            
            self.export_button.setEnabled(True)
            self.log_text.append("Model loaded successfully! You can now export it to different formats.")
            QMessageBox.information(
                self,
                "Success",
                f"Model loaded successfully!\n\nArchitecture: {model_arch}\nClasses: {num_classes}"
            )
            
        except Exception as e:
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}\n\nSee log for details.")
            self.log_text.append(f"Error loading model: {str(e)}")
            self.log_text.append(f"Full traceback:\n{error_details}")
    
    def export_model(self):
        if not self.training_thread or not hasattr(self.training_thread, 'trained_model') or self.training_thread.trained_model is None:
            QMessageBox.warning(self, "No Model", "No trained model available. Please train a model first.")
            return
        
        # Show format selection dialog
        format_dialog = QDialog(self)
        format_dialog.setWindowTitle("Export Model")
        format_dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Format selection
        format_label = QLabel("Select export format:")
        layout.addWidget(format_label)
        
        format_group = QButtonGroup(format_dialog)
        
        # PyTorch format (default)
        pth_radio = QRadioButton("PyTorch (.pth) - Full model with metadata")
        pth_radio.setChecked(True)
        format_group.addButton(pth_radio, 0)
        layout.addWidget(pth_radio)
        
        # TorchScript format
        pt_radio = QRadioButton("TorchScript (.pt) - Optimized for deployment")
        format_group.addButton(pt_radio, 1)
        layout.addWidget(pt_radio)
        
        # ONNX format
        onnx_radio = QRadioButton("ONNX (.onnx) - Cross-platform inference")
        format_group.addButton(onnx_radio, 2)
        layout.addWidget(onnx_radio)
        
        # State dict only
        state_radio = QRadioButton("State Dict Only (.pth) - Weights only")
        format_group.addButton(state_radio, 3)
        layout.addWidget(state_radio)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(format_dialog.accept)
        cancel_button.clicked.connect(format_dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        format_dialog.setLayout(layout)
        
        if format_dialog.exec_() != QDialog.Accepted:
            return
        
        # Get selected format
        selected_format = format_group.checkedId()
        
        # Set up file dialog based on format
        if selected_format == 0:  # PyTorch full
            filter_str = "PyTorch Model (*.pth)"
            default_name = "trained_model.pth"
        elif selected_format == 1:  # TorchScript
            filter_str = "TorchScript Model (*.pt)"
            default_name = "trained_model.pt"
        elif selected_format == 2:  # ONNX
            filter_str = "ONNX Model (*.onnx)"
            default_name = "trained_model.onnx"
        else:  # State dict only
            filter_str = "PyTorch State Dict (*.pth)"
            default_name = "model_state_dict.pth"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            default_name,
            filter_str
        )
        
        if not filename:
            return
        
        try:
            model = self.training_thread.trained_model
            metadata = self.training_thread.model_metadata
            
            if selected_format == 0:  # PyTorch full
                torch.save({
                    'model_state_dict': model.state_dict(),
                    **metadata
                }, filename)
                self.log_text.append(f"Model exported as PyTorch format to: {filename}")
                
            elif selected_format == 1:  # TorchScript
                model.eval()
                # Get input shape from model metadata
                num_classes = metadata['num_classes']
                example_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
                traced_model = torch.jit.trace(model, example_input)
                torch.jit.save(traced_model, filename)
                
                # Save metadata separately
                metadata_file = filename.replace('.pt', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                self.log_text.append(f"Model exported as TorchScript to: {filename}")
                self.log_text.append(f"Metadata saved to: {metadata_file}")
                
            elif selected_format == 2:  # ONNX
                try:
                    import onnx
                except ImportError:
                    QMessageBox.critical(
                        self, 
                        "ONNX Not Available", 
                        "ONNX package is not installed. Please install it using:\n\npip install onnx"
                    )
                    return
                
                model.eval()
                device = next(model.parameters()).device
                example_input = torch.randn(1, 3, 224, 224).to(device)
                
                self.log_text.append("Exporting to ONNX format...")
                
                torch.onnx.export(
                    model,
                    example_input,
                    filename,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
                
                # Save metadata separately
                metadata_file = filename.replace('.onnx', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                self.log_text.append(f"Model exported as ONNX to: {filename}")
                self.log_text.append(f"Metadata saved to: {metadata_file}")
                
            else:  # State dict only
                torch.save(model.state_dict(), filename)
                
                # Save metadata separately
                metadata_file = filename.replace('.pth', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                self.log_text.append(f"Model state dict exported to: {filename}")
                self.log_text.append(f"Metadata saved to: {metadata_file}")
            
            QMessageBox.information(self, "Success", f"Model exported successfully to:\n{filename}")
            
        except Exception as e:
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export model:\n{str(e)}\n\nSee log for details.")
            self.log_text.append(f"Error exporting model: {str(e)}")
            self.log_text.append(f"Full traceback:\n{error_details}")
    
    def export_stats(self):
        """Export training statistics and visualizations"""
        if not self.training_thread or not hasattr(self.training_thread, 'test_predictions'):
            QMessageBox.warning(self, "No Statistics", "No training statistics available. Please train a model first.")
            return
        
        if len(self.training_thread.test_predictions) == 0:
            QMessageBox.warning(self, "No Statistics", "No test predictions available. Please train a model first.")
            return
        
        # Select directory to save stats
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export Statistics",
            ""
        )
        
        if not directory:
            return
        
        try:
            self.log_text.append(f"Exporting statistics to: {directory}")
            
            # Get data from training thread
            test_predictions = self.training_thread.test_predictions
            test_labels = self.training_thread.test_labels
            test_probabilities = self.training_thread.test_probabilities
            train_accuracies = self.training_thread.train_accuracies
            val_accuracies = self.training_thread.val_accuracies
            train_losses = self.training_thread.train_losses
            val_losses = self.training_thread.val_losses
            metadata = self.training_thread.model_metadata
            classes = metadata.get('classes', [f"Class {i}" for i in range(metadata.get('num_classes', 10))])
            num_classes = len(classes)
            
            # Set style for plots
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (10, 8)
            
            # 1. Accuracy plot (Training and Validation)
            if len(val_accuracies) > 0 and len(train_accuracies) > 0:
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(val_accuracies) + 1)
                plt.plot(epochs, train_accuracies, 'r-', linewidth=2, label='Training Accuracy')
                plt.plot(epochs, val_accuracies, 'b-', linewidth=2, label='Validation Accuracy')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Training and Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                accuracy_path = os.path.join(directory, 'accuracy.png')
                plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.log_text.append(f"Saved accuracy plot to: {accuracy_path}")
            
            # 2. Loss plot
            if len(train_losses) > 0 and len(val_losses) > 0:
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(train_losses) + 1)
                plt.plot(epochs, train_losses, 'r-', linewidth=2, label='Training Loss')
                plt.plot(epochs, val_losses, 'b-', linewidth=2, label='Validation Loss')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                loss_path = os.path.join(directory, 'loss.png')
                plt.savefig(loss_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.log_text.append(f"Saved loss plot to: {loss_path}")
            
            # 3. Confusion Matrix
            cm = confusion_matrix(test_labels, test_predictions)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes,
                       cbar_kws={'label': 'Count'})
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            cm_path = os.path.join(directory, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log_text.append(f"Saved confusion matrix to: {cm_path}")
            
            # 4. Precision, Recall, F1-Score per class
            precision = precision_score(test_labels, test_predictions, average=None, zero_division=0)
            recall = recall_score(test_labels, test_predictions, average=None, zero_division=0)
            f1 = f1_score(test_labels, test_predictions, average=None, zero_division=0)
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(classes))
            width = 0.25
            
            plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
            plt.bar(x, recall, width, label='Recall', alpha=0.8)
            plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
            
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Precision, Recall, and F1-Score per Class', fontsize=14, fontweight='bold')
            plt.xticks(x, classes, rotation=45, ha='right')
            plt.legend(fontsize=10)
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            metrics_path = os.path.join(directory, 'precision_recall_f1.png')
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log_text.append(f"Saved precision/recall/F1 plot to: {metrics_path}")
            
            # 5. Individual Precision plot
            plt.figure(figsize=(12, 6))
            x = np.arange(len(classes))
            plt.bar(x, precision, color='#FF6B6B', alpha=0.8)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision per Class', fontsize=14, fontweight='bold')
            plt.xticks(x, classes, rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            precision_path = os.path.join(directory, 'precision.png')
            plt.savefig(precision_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log_text.append(f"Saved precision plot to: {precision_path}")
            
            # 6. Individual Recall plot
            plt.figure(figsize=(12, 6))
            plt.bar(x, recall, color='#4ECDC4', alpha=0.8)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Recall', fontsize=12)
            plt.title('Recall per Class', fontsize=14, fontweight='bold')
            plt.xticks(x, classes, rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            recall_path = os.path.join(directory, 'recall.png')
            plt.savefig(recall_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log_text.append(f"Saved recall plot to: {recall_path}")
            
            # 7. Individual F1-Score plot
            plt.figure(figsize=(12, 6))
            plt.bar(x, f1, color='#45B7D1', alpha=0.8)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('F1-Score', fontsize=12)
            plt.title('F1-Score per Class', fontsize=14, fontweight='bold')
            plt.xticks(x, classes, rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            f1_path = os.path.join(directory, 'f1_score.png')
            plt.savefig(f1_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log_text.append(f"Saved F1-score plot to: {f1_path}")
            
            # 8. Overall metrics summary
            avg_precision = precision_score(test_labels, test_predictions, average='macro', zero_division=0)
            avg_recall = recall_score(test_labels, test_predictions, average='macro', zero_division=0)
            avg_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
            
            plt.figure(figsize=(8, 6))
            metrics_names = ['Precision', 'Recall', 'F1-Score']
            metrics_values = [avg_precision, avg_recall, avg_f1]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
            plt.ylabel('Score', fontsize=12)
            plt.title('Overall Test Set Metrics (Macro Average)', fontsize=14, fontweight='bold')
            plt.ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            summary_path = os.path.join(directory, 'overall_metrics.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log_text.append(f"Saved overall metrics to: {summary_path}")
            
            # 7. Save metrics as JSON
            metrics_dict = {
                'model_architecture': metadata.get('model_architecture', 'Unknown'),
                'test_accuracy': float(metadata.get('test_accuracy', 0.0)),
                'best_val_accuracy': float(metadata.get('best_val_accuracy', 0.0)),
                'precision_macro': float(avg_precision),
                'recall_macro': float(avg_recall),
                'f1_macro': float(avg_f1),
                'per_class_metrics': {
                    classes[i]: {
                        'precision': float(precision[i]),
                        'recall': float(recall[i]),
                        'f1_score': float(f1[i])
                    }
                    for i in range(len(classes))
                }
            }
            
            metrics_json_path = os.path.join(directory, 'metrics.json')
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            self.log_text.append(f"Saved metrics JSON to: {metrics_json_path}")
            
            QMessageBox.information(
                self,
                "Success",
                f"Statistics exported successfully to:\n{directory}\n\n"
                f"Generated files:\n"
                f"- accuracy.png (Training & Validation)\n"
                f"- loss.png\n"
                f"- confusion_matrix.png\n"
                f"- precision_recall_f1.png\n"
                f"- precision.png\n"
                f"- recall.png\n"
                f"- f1_score.png\n"
                f"- overall_metrics.png\n"
                f"- metrics.json"
            )
            
        except Exception as e:
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export statistics:\n{str(e)}\n\nSee log for details.")
            self.log_text.append(f"Error exporting statistics: {str(e)}")
            self.log_text.append(f"Full traceback:\n{error_details}")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = ChipTrainerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
