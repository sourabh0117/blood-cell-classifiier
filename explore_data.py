# explore_data.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(data_dir='data'):
    """
    Analyzes the dataset by counting images per class and visualizing the distribution.
    """
    print("--- Dataset Analysis ---")
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not class_names:
        print(f"Error: No subdirectories found in '{data_dir}'. Please ensure your data is structured correctly.")
        return

    print(f"Found {len(class_names)} classes: {class_names}")

    data = []
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path) if f.endswith(('jpg', 'jpeg', 'png'))])
        data.append({'Class': class_name, 'Count': num_images})

    df = pd.DataFrame(data)

    print("\nImage distribution per class:")
    print(df.to_string(index=False))

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Class', data=df, palette='viridis', hue='Class', dodge=False, legend=False)
    plt.title('Distribution of Blood Cell Images per Class')
    plt.xlabel('Number of Images')
    plt.ylabel('Class')

    # Add counts to the bars
    for index, value in enumerate(df['Count']):
        plt.text(value, index, f' {value}', va='center')

    plt.tight_layout()

    # Save the plot
    output_dir = 'outputs/results'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(plot_path)
    print(f"\nClass distribution plot saved to: {plot_path}")
    print("------------------------")


if __name__ == "__main__":
    analyze_dataset()