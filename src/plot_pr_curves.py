# src/plot_pr_curves.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse
import os

# --- Import project modules ---
from src import config

def plot_precision_recall_curves(model_name):
    """
    Loads true labels and prediction scores to plot per-class Precision-Recall curves.
    """
    print(f"--- Generating Precision-Recall Curves for model: {model_name} ---")

    # --- 1. Load the saved data ---
    results_dir = os.path.join(config.OUTPUT_DIR, 'results')
    labels_path = os.path.join(results_dir, f'{model_name}_true_labels.npy')
    scores_path = os.path.join(results_dir, f'{model_name}_scores.npy')

    if not os.path.exists(labels_path) or not os.path.exists(scores_path):
        print(f"Error: Data files not found. Please run 'evaluate.py' first to generate them.")
        return

    y_true = np.load(labels_path)
    y_scores = np.load(scores_path)
    
    # --- 2. Plot PR curve for each class ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Use a color cycle
    colors = plt.cm.get_cmap('viridis', config.NUM_CLASSES)

    for i in range(config.NUM_CLASSES):
        # Create a binary version of the labels for the current class (one-vs-rest)
        y_true_class = (y_true == i).astype(int)
        
        # Get the scores for the current class
        y_scores_class = y_scores[:, i]
        
        # Calculate precision, recall, and average precision
        precision, recall, _ = precision_recall_curve(y_true_class, y_scores_class)
        avg_precision = average_precision_score(y_true_class, y_scores_class)
        
        class_name = config.CLASS_NAMES[i]
        plt.plot(recall, precision, color=colors(i), lw=2,
                 label=f'{class_name} (AP = {avg_precision:.2f})')

    # --- 3. Finalize and save the plot ---
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Figure 8: Per-Class Precision-Recall Curves', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True)
    
    save_path = os.path.join(results_dir, f'{model_name}_pr_curve.png')
    plt.savefig(save_path)
    print(f"✅ Precision-Recall curve plot saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Precision-Recall curves from evaluation data.")
    parser.add_argument(
        '--model', type=str, required=True, 
        choices=['resnet50', 'efficientnet_b0', 'densenet121'],
        help='The model name corresponding to the saved data files.'
    )
    args = parser.parse_args()
    
    plot_precision_recall_curves(model_name=args.model)