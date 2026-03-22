# # src/evaluate.py

# import torch
# from torch.utils.data import DataLoader
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import glob
# import os
# import argparse

# # --- Import project modules ---
# from src import config
# from src.data_utils import dataset, augmentations
# from src.models import cnn_models

# def run_evaluation(model_name):
#     """
#     Loads all saved models from a k-fold cross-validation run, evaluates them
#     on their respective validation sets, and aggregates the results.
#     """
#     print(f"--- Starting Evaluation for model: {model_name} ---")

#     # --- 1. Load Data (same split as training) ---
#     all_filepaths, all_labels, class_to_idx = train.get_all_filepaths_and_labels()
#     skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)

#     all_true_labels = []
#     all_predictions = []
#     fold_accuracies = []

#     # --- 2. Loop through each fold ---
#     for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels)):
#         print(f"\n--- Evaluating Fold {fold + 1}/{config.NUM_FOLDS} ---")
        
#         # --- Find the model for this fold ---
#         model_path = os.path.join(config.OUTPUT_DIR, 'models', f'{model_name}_fold_{fold+1}_best_augmented.pth')
#         if not os.path.exists(model_path):
#             print(f"Warning: Model not found for fold {fold+1} at {model_path}. Skipping.")
#             continue
        
#         # --- Load Model ---
#         model_func = getattr(cnn_models, f"get_{model_name}")
#         model = model_func(pretrained=False) # No need to load pretrained weights, we load our own
#         model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
#         model.to(config.DEVICE)
#         model.eval()

#         # --- Create Validation DataLoader for this fold ---
#         val_filepaths = [all_filepaths[i] for i in val_idx]
#         val_labels = [all_labels[i] for i in val_idx]
        
#         val_dataset = dataset.BloodCellDataset(
#             image_paths=val_filepaths, labels=val_labels,
#             class_to_idx=class_to_idx, transforms=augmentations.get_val_transforms()
#         )
#         val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
#         fold_preds = []
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
                
#                 fold_preds.extend(preds.cpu().numpy())
#                 all_true_labels.extend(labels.cpu().numpy())
        
#         all_predictions.extend(fold_preds)
        
#         # Calculate and store accuracy for this fold
#         fold_acc = accuracy_score(val_labels, fold_preds)
#         fold_accuracies.append(fold_acc)
#         print(f"Fold {fold+1} Validation Accuracy: {fold_acc:.4f}")

#     # --- 3. Statistical Analysis ---
#     print("\n" + "="*50)
#     print("--- Statistical Analysis (Cross-Validation Results) ---")
#     mean_acc = np.mean(fold_accuracies)
#     std_acc = np.std(fold_accuracies)
#     print(f"Validation Accuracies per Fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
#     print(f"Mean Validation Accuracy: {mean_acc:.4f}")
#     print(f"Standard Deviation of Accuracy: {std_acc:.4f}")
#     print(f"Final Reported Performance: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
#     # --- 4. Aggregated Classification Report ---
#     print("\n" + "="*50)
#     print("--- Aggregated Classification Report ---")
#     # We use idx_to_class to get the names for the report
#     idx_to_class = {v: k for k, v in class_to_idx.items()}
#     class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
#     report = classification_report(all_true_labels, all_predictions, target_names=class_names)
#     print(report)

#     # --- 5. Aggregated Confusion Matrix ---
#     print("\n" + "="*50)
#     print("--- Aggregated Confusion Matrix ---")
#     cm = confusion_matrix(all_true_labels, all_predictions)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title(f'Aggregated Confusion Matrix for {model_name}')
    
#     # Save the plot
#     save_path = os.path.join(config.OUTPUT_DIR, 'results', f'{model_name}_confusion_matrix.png')
#     plt.savefig(save_path)
#     print(f"Confusion matrix saved to {save_path}")
#     print("="*50)

# if __name__ == '__main__':
#     # Add a temporary import of train to use its utility function
#     from src import train
    
#     parser = argparse.ArgumentParser(description="Evaluate a trained CNN model.")
#     parser.add_argument(
#         '--model', type=str, required=True, 
#         choices=['resnet50', 'efficientnet_b0', 'densenet121'],
#         help='The model architecture to evaluate.'
#     )
#     args = parser.parse_args()
    
#     run_evaluation(model_name=args.model)





# src/evaluate.py (FINAL COMPLETE VERSION)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# --- Import project modules ---
from src import config
from src.data_utils import dataset, augmentations
from src.models import cnn_models
from src import train # Use the get_all_filepaths_and_labels function from train.py

def run_evaluation(model_name):
    """
    Loads all saved models from a k-fold cross-validation run, evaluates them,
    aggregates the results, and saves scores for PR curve plotting.
    """
    print(f"--- Starting Evaluation for model: {model_name} ---")

    # --- 1. Load Data (same split as training) ---
    all_filepaths, all_labels, class_to_idx = train.get_all_filepaths_and_labels()
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)

    all_true_labels = []
    all_predictions = []
    all_scores = [] # <-- ADDED: To store probability scores
    fold_accuracies = []

    # --- 2. Loop through each fold ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels)):
        print(f"\n--- Evaluating Fold {fold + 1}/{config.NUM_FOLDS} ---")
        
        model_path = os.path.join(config.OUTPUT_DIR, 'models', f'{model_name}_fold_{fold+1}_best_augmented.pth')
        if not os.path.exists(model_path):
            print(f"Warning: Model not found for fold {fold+1} at {model_path}. Skipping.")
            continue
        
        model_func = getattr(cnn_models, f"get_{model_name}")
        model = model_func(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()

        val_filepaths = [all_filepaths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        val_dataset = dataset.BloodCellDataset(
            image_paths=val_filepaths, labels=val_labels,
            class_to_idx=class_to_idx, transforms=augmentations.get_val_transforms()
        )
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        fold_preds = []
        softmax = nn.Softmax(dim=1) # <-- ADDED: Softmax to convert logits to probabilities

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config.DEVICE)
                labels_cpu = labels.cpu().numpy()
                
                outputs = model(inputs)
                scores = softmax(outputs).cpu().numpy() # <-- MODIFIED: Get scores
                _, preds = torch.max(outputs, 1)
                preds_cpu = preds.cpu().numpy()
                
                fold_preds.extend(preds_cpu)
                all_true_labels.extend(labels_cpu)
                all_scores.extend(scores) # <-- ADDED: Save scores
        
        all_predictions.extend(fold_preds)
        
        fold_acc = accuracy_score(val_labels, fold_preds)
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold+1} Validation Accuracy: {fold_acc:.4f}")

    # --- 3. Save Scores and Labels for Plotting ---
    print("\n" + "="*50)
    print("--- Saving data for external plots ---")
    results_dir = os.path.join(config.OUTPUT_DIR, 'results')
    labels_save_path = os.path.join(results_dir, f'{model_name}_true_labels.npy')
    scores_save_path = os.path.join(results_dir, f'{model_name}_scores.npy')

    np.save(labels_save_path, np.array(all_true_labels))
    np.save(scores_save_path, np.array(all_scores))
    print(f"True labels saved to: {labels_save_path}")
    print(f"Prediction scores saved to: {scores_save_path}")

    # --- 4. Statistical Analysis ---
    print("\n" + "="*50)
    print("--- Statistical Analysis (Cross-Validation Results) ---")
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"Mean Validation Accuracy: {mean_acc:.4f}")
    print(f"Standard Deviation of Accuracy: {std_acc:.4f}")
    
    # --- 5. Aggregated Classification Report ---
    print("\n" + "="*50)
    print("--- Aggregated Classification Report ---")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(all_true_labels, all_predictions, target_names=class_names)
    print(report)

    # --- 6. Aggregated Confusion Matrix ---
    print("\n" + "="*50)
    print("--- Aggregated Confusion Matrix ---")
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Aggregated Confusion Matrix for {model_name}')
    
    save_path = os.path.join(results_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model.")
    parser.add_argument(
        '--model', type=str, required=True, 
        choices=['resnet50', 'efficientnet_b0', 'densenet121'],
        help='The model architecture to evaluate.'
    )
    args = parser.parse_args()
    
    run_evaluation(model_name=args.model)