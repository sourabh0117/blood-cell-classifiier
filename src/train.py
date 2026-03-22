# # # src/train.py

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader, Subset
# # import torch.optim as optim
# # from sklearn.model_selection import StratifiedKFold
# # from sklearn.metrics import accuracy_score
# # import numpy as np
# # import glob
# # import os
# # import argparse
# # from tqdm import tqdm
# # import time

# # # --- Import project modules ---
# # from src import config
# # from src.data_utils import dataset, augmentations
# # from src.models import cnn_models

# # def get_all_filepaths_and_labels():
# #     """Scans the data directory and returns a list of all image filepaths and their corresponding labels."""
# #     filepaths = []
# #     labels = []

# #     # Create a mapping from class name to integer index
# #     class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}

# #     for class_name in config.CLASS_NAMES:
# #         class_dir = os.path.join(config.DATA_DIR, class_name)
# #         # Using glob to find all jpg images
# #         class_filepaths = glob.glob(os.path.join(class_dir, '*.jpg'))
# #         filepaths.extend(class_filepaths)
# #         labels.extend([class_to_idx[class_name]] * len(class_filepaths))

# #     return filepaths, labels, class_to_idx


# # def create_sampler(labels):
# #     """Creates a WeightedRandomSampler to handle class imbalance."""
# #     class_counts = np.bincount(labels)
# #     class_weights = 1. / class_counts
# #     sample_weights = np.array([class_weights[label] for label in labels])
# #     sampler = torch.utils.data.WeightedRandomSampler(
# #         weights=sample_weights,
# #         num_samples=len(sample_weights),
# #         replacement=True
# #     )
# #     return sampler


# # def train_one_epoch(model, dataloader, criterion, optimizer, device):
# #     """Trains the model for one epoch."""
# #     model.train()
# #     running_loss = 0.0
# #     all_preds = []
# #     all_labels = []

# #     progress_bar = tqdm(dataloader, desc="Training", unit="batch")
# #     for inputs, labels in progress_bar:
# #         inputs, labels = inputs.to(device), labels.to(device)

# #         # Zero the parameter gradients
# #         optimizer.zero_grad()

# #         # Forward pass
# #         outputs = model(inputs)
# #         loss = criterion(outputs, labels)

# #         # Backward pass and optimize
# #         loss.backward()
# #         optimizer.step()

# #         running_loss += loss.item() * inputs.size(0)

# #         # For accuracy calculation
# #         _, preds = torch.max(outputs, 1)
# #         all_preds.extend(preds.cpu().numpy())
# #         all_labels.extend(labels.cpu().numpy())

# #         progress_bar.set_postfix(loss=loss.item())

# #     epoch_loss = running_loss / len(dataloader.dataset)
# #     epoch_acc = accuracy_score(all_labels, all_preds)
# #     return epoch_loss, epoch_acc


# # def validate_one_epoch(model, dataloader, criterion, device):
# #     """Validates the model for one epoch."""
# #     model.eval()
# #     running_loss = 0.0
# #     all_preds = []
# #     all_labels = []

# #     with torch.no_grad():
# #         progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
# #         for inputs, labels in progress_bar:
# #             inputs, labels = inputs.to(device), labels.to(device)

# #             outputs = model(inputs)
# #             loss = criterion(outputs, labels)

# #             running_loss += loss.item() * inputs.size(0)

# #             _, preds = torch.max(outputs, 1)
# #             all_preds.extend(preds.cpu().numpy())
# #             all_labels.extend(labels.cpu().numpy())
# #             progress_bar.set_postfix(loss=loss.item())

# #     epoch_loss = running_loss / len(dataloader.dataset)
# #     epoch_acc = accuracy_score(all_labels, all_preds)
# #     return epoch_loss, epoch_acc


# # def run_training(model_name):
# #     """Main function to run the k-fold cross-validation training."""

# #     # --- 1. Data Preparation ---
# #     print("Loading all filepaths and labels...")
# #     all_filepaths, all_labels, class_to_idx = get_all_filepaths_and_labels()

# #     # --- 2. Stratified K-Fold Setup ---
# #     skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)

# #     for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels)):
# #         print("-" * 50)
# #         print(f"========== FOLD {fold + 1}/{config.NUM_FOLDS} ==========")
# #         print("-" * 50)

# #         # --- 3. Create Datasets and DataLoaders for the current fold ---
# #         train_filepaths = [all_filepaths[i] for i in train_idx]
# #         train_labels = [all_labels[i] for i in train_idx]
# #         val_filepaths = [all_filepaths[i] for i in val_idx]
# #         val_labels = [all_labels[i] for i in val_idx]

# #         train_dataset = dataset.BloodCellDataset(
# #             image_paths=train_filepaths,
# #             labels=train_labels,
# #             class_to_idx=class_to_idx,
# #             transforms=augmentations.get_train_transforms()
# #         )
# #         val_dataset = dataset.BloodCellDataset(
# #             image_paths=val_filepaths,
# #             labels=val_labels,
# #             class_to_idx=class_to_idx,
# #             transforms=augmentations.get_val_transforms()
# #         )

# #         train_sampler = create_sampler(train_labels)

# #         train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
# #         val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# #         # --- 4. Model, Criterion, Optimizer Setup ---
# #         print(f"Initializing model: {model_name}")
# #         if model_name == 'resnet50':
# #             model = cnn_models.get_resnet50()
# #         elif model_name == 'efficientnet_b0':
# #             model = cnn_models.get_efficientnet_b0()
# #         elif model_name == 'densenet121':
# #             model = cnn_models.get_densenet121()
# #         else:
# #             raise ValueError(f"Model {model_name} not recognized.")

# #         model.to(config.DEVICE)

# #         criterion = nn.CrossEntropyLoss()
# #         optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# #         # --- 5. Training Loop ---
# #         best_val_acc = 0.0

# #         for epoch in range(config.NUM_EPOCHS):
# #             print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

# #             start_time = time.time()

# #             train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
# #             val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, config.DEVICE)

# #             end_time = time.time()
# #             epoch_duration = end_time - start_time

# #             print(f"Epoch {epoch + 1} Summary | Duration: {epoch_duration:.2f}s")
# #             print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
# #             print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")

# #             # Save the best model
# #             if val_acc > best_val_acc:
# #                 best_val_acc = val_acc
# #                 save_path = os.path.join(config.OUTPUT_DIR, 'models', f'{model_name}_fold_{fold+1}_best.pth')
# #                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
# #                 torch.save(model.state_dict(), save_path)
# #                 print(f"✨ New best model saved to {save_path} with validation accuracy: {best_val_acc:.4f}")

# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description="Train a CNN for blood cell classification.")
# #     parser.add_argument(
# #         '--model', 
# #         type=str, 
# #         required=True, 
# #         choices=['resnet50', 'efficientnet_b0', 'densenet121'],
# #         help='The model architecture to train.'
# #     )
# #     args = parser.parse_args()

# #     run_training(model_name=args.model)




# # src/train.py (REVISED VERSION)

# # src/train.py (FULLY REVISED IMPORTS)

# import torch
# import torch.nn as nn
# import torch.optim as optim  # <-- This was the missing line
# from torch.utils.data import DataLoader
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np
# import glob
# import os
# import argparse
# from tqdm import tqdm
# import time

# # --- Import project modules ---
# from src import config
# from src.data_utils import dataset, augmentations
# from src.models import cnn_models

# # In src/train.py, replace the old function with this new one:

# def get_all_filepaths_and_labels():
#     """Scans the data directory and returns a list of all image filepaths and their corresponding labels."""
    
#     # --- START OF DEBUG BLOCK ---
#     print("="*50)
#     print(f"DEBUG: Attempting to read from data directory: '{config.DATA_DIR}'")
#     # --- END OF DEBUG BLOCK ---

#     filepaths = []
#     labels = []
#     class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    
#     for class_name in config.CLASS_NAMES:
#         class_dir = os.path.join(config.DATA_DIR, class_name)
#         if not os.path.isdir(class_dir):
#             print(f"DEBUG: WARNING - Directory not found: {class_dir}")
#             continue
        
#         class_filepaths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#         filepaths.extend(class_filepaths)
#         labels.extend([class_to_idx[class_name]] * len(class_filepaths))
            
#     # --- START OF DEBUG BLOCK ---
#     print(f"DEBUG: Successfully found a total of {len(filepaths)} image files.")
#     if len(filepaths) == 0:
#         print("DEBUG: CRITICAL ERROR! No images were found. Please check that the path in 'src/config.py' is correct and the folder contains images.")
#     print("="*50)
#     # --- END OF DEBUG BLOCK ---

#     return filepaths, labels, class_to_idx
# # Sampler function is no longer needed because the dataset is balanced
# # def create_sampler(labels): ... (REMOVED)

# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     """Trains the model for one epoch."""
#     model.train()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     progress_bar = tqdm(dataloader, desc="Training", unit="batch")
#     for inputs, labels in progress_bar:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)
#         _, preds = torch.max(outputs, 1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         progress_bar.set_postfix(loss=loss.item())

#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     return epoch_loss, epoch_acc

# def validate_one_epoch(model, dataloader, criterion, device):
#     """Validates the model for one epoch."""
#     model.eval()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
#         for inputs, labels in progress_bar:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             progress_bar.set_postfix(loss=loss.item())

#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     return epoch_loss, epoch_acc

# def run_training(model_name):
#     """Main function to run the k-fold cross-validation training."""
    
#     print("Loading all filepaths and labels from the augmented dataset...")
#     all_filepaths, all_labels, class_to_idx = get_all_filepaths_and_labels()
    
#     skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    
#     for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels)):
#         print("-" * 50)
#         print(f"========== FOLD {fold + 1}/{config.NUM_FOLDS} ==========")
#         print("-" * 50)
        
#         train_filepaths = [all_filepaths[i] for i in train_idx]
#         train_labels = [all_labels[i] for i in train_idx]
#         val_filepaths = [all_filepaths[i] for i in val_idx]
#         val_labels = [all_labels[i] for i in val_idx]

#         train_dataset = dataset.BloodCellDataset(
#             image_paths=train_filepaths, labels=train_labels,
#             class_to_idx=class_to_idx, transforms=augmentations.get_train_transforms()
#         )
#         val_dataset = dataset.BloodCellDataset(
#             image_paths=val_filepaths, labels=val_labels,
#             class_to_idx=class_to_idx, transforms=augmentations.get_val_transforms()
#         )
        
#         # --- KEY CHANGE: No sampler, just shuffle the training data ---
#         train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
#         print(f"Initializing model: {model_name}")
#         model_func = getattr(cnn_models, f"get_{model_name}")
#         model = model_func()
#         model.to(config.DEVICE)
        
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        
#         best_val_acc = 0.0
        
#         for epoch in range(config.NUM_EPOCHS):
#             print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
            
#             start_time = time.time()
#             train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
#             val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
#             end_time = time.time()
            
#             print(f"Epoch {epoch + 1} Summary | Duration: {end_time - start_time:.2f}s")
#             print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#             print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")
            
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 save_path = os.path.join(config.OUTPUT_DIR, 'models', f'{model_name}_fold_{fold+1}_best_augmented.pth')
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                 torch.save(model.state_dict(), save_path)
#                 print(f"✨ New best model saved to {save_path} with validation accuracy: {best_val_acc:.4f}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train a CNN for blood cell classification.")
#     parser.add_argument(
#         '--model', type=str, required=True, 
#         choices=['resnet50', 'efficientnet_b0', 'densenet121'],
#         help='The model architecture to train.'
#     )
#     args = parser.parse_args()
    
#     run_training(model_name=args.model)





# src/train.py (FINAL COMPLETE VERSION)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import time
import pandas as pd

# --- Import project modules ---
from src import config
from src.data_utils import dataset, augmentations
from src.models import cnn_models

def get_all_filepaths_and_labels():
    """Scans the data directory and returns a list of all image filepaths and their corresponding labels."""
    
    # --- START OF DEBUG BLOCK ---
    print("="*50)
    print(f"DEBUG: Attempting to read from data directory: '{config.DATA_DIR}'")
    # --- END OF DEBUG BLOCK ---

    filepaths = []
    labels = []
    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"DEBUG: WARNING - Directory not found: {class_dir}")
            continue
        
        class_filepaths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        filepaths.extend(class_filepaths)
        labels.extend([class_to_idx[class_name]] * len(class_filepaths))
            
    # --- START OF DEBUG BLOCK ---
    print(f"DEBUG: Successfully found a total of {len(filepaths)} image files.")
    if len(filepaths) == 0:
        print("DEBUG: CRITICAL ERROR! No images were found. Please check that the path in 'src/config.py' is correct and the folder contains images.")
    print("="*50)
    # --- END OF DEBUG BLOCK ---

    return filepaths, labels, class_to_idx

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """Validates the model for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def run_training(model_name, target_fold=None):
    """Main function to run the k-fold cross-validation training."""
    
    print("Loading all filepaths and labels from the augmented dataset...")
    all_filepaths, all_labels, class_to_idx = get_all_filepaths_and_labels()
    
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels)):
        # If a specific fold is targeted, skip others
        if target_fold is not None and fold + 1 != target_fold:
            continue

        print("-" * 50)
        print(f"========== FOLD {fold + 1}/{config.NUM_FOLDS} ==========")
        print("-" * 50)
        
        fold_history = []
        
        train_filepaths = [all_filepaths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_filepaths = [all_filepaths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]

        train_dataset = dataset.BloodCellDataset(
            image_paths=train_filepaths, labels=train_labels,
            class_to_idx=class_to_idx, transforms=augmentations.get_train_transforms()
        )
        val_dataset = dataset.BloodCellDataset(
            image_paths=val_filepaths, labels=val_labels,
            class_to_idx=class_to_idx, transforms=augmentations.get_val_transforms()
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print(f"Initializing model: {model_name}")
        model_func = getattr(cnn_models, f"get_{model_name}")
        model = model_func()
        model.to(config.DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        
        best_val_acc = 0.0
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
            
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
            end_time = time.time()
            
            print(f"Epoch {epoch + 1} Summary | Duration: {end_time - start_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")

            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(config.OUTPUT_DIR, 'models', f'{model_name}_fold_{fold+1}_best_augmented.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✨ New best model saved to {save_path} with validation accuracy: {best_val_acc:.4f}")

        history_df = pd.DataFrame(fold_history)
        history_save_path = os.path.join(config.OUTPUT_DIR, 'results', f'{model_name}_fold_{fold+1}_history.csv')
        os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
        history_df.to_csv(history_save_path, index=False)
        print(f"📈 Fold {fold+1} history saved to {history_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN for blood cell classification.")
    parser.add_argument(
        '--model', type=str, required=True, 
        choices=['resnet50', 'efficientnet_b0', 'densenet121'],
        help='The model architecture to train.'
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        choices=range(1, config.NUM_FOLDS + 1),
        help=f'Specify a single fold to train (from 1 to {config.NUM_FOLDS}). If not specified, all folds will be trained.'
    )
    args = parser.parse_args()
    
    run_training(model_name=args.model, target_fold=args.fold)