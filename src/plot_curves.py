import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_learning_curves(history_path, save_dir):
    """
    Reads a training history CSV and plots the learning curves for loss and accuracy
    into two separate image files.
    """
    print(f"Reading history from: {history_path}")
    if not os.path.exists(history_path):
        print(f"Error: History file not found at {history_path}")
        return

    history_df = pd.read_csv(history_path)

    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # --- Figure 5: Training & Validation Loss Curves ---
    plt.figure(figsize=(10, 6)) # Create the first figure for Loss
    plt.plot(history_df['epoch'], history_df['train_loss'], 'o-', label='Training Loss', color='royalblue')
    plt.plot(history_df['epoch'], history_df['val_loss'], 'o-', label='Validation Loss', color='orangered')
    plt.title('Figure 5: Training & Validation Loss per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Determine save path for the loss plot
    base_name = os.path.basename(history_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    loss_save_path = os.path.join(save_dir, f'{file_name_no_ext}_loss_curve.png')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(loss_save_path)
    print(f"✅ Figure 5 (Loss Curves) saved to: {loss_save_path}")
    
    # --- Figure 6: Training & Validation Accuracy Curves ---
    plt.figure(figsize=(10, 6)) # Create the second, separate figure for Accuracy
    plt.plot(history_df['epoch'], history_df['train_acc'], 'o-', label='Training Accuracy', color='forestgreen')
    plt.plot(history_df['epoch'], history_df['val_acc'], 'o-', label='Validation Accuracy', color='purple')
    plt.title('Figure 6: Training & Validation Accuracy per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Determine save path for the accuracy plot
    accuracy_save_path = os.path.join(save_dir, f'{file_name_no_ext}_accuracy_curve.png')
    plt.savefig(accuracy_save_path)
    print(f"✅ Figure 6 (Accuracy Curves) saved to: {accuracy_save_path}")
    
    # Optional: Display the plots if running in an interactive environment
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot learning curves from a training history file into separate plots.")
    parser.add_argument(
        '--history-path', 
        type=str, 
        required=True,
        help='Path to the fold history CSV file (e.g., outputs/results/resnet50_fold_1_history.csv).'
    )
    args = parser.parse_args()
    
    plot_learning_curves(history_path=args.history_path, save_dir='outputs/results')