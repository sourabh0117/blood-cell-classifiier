# # create_augmented_dataset.py

# import os
# import glob
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm
# import random

# # --- Configuration ---
# SOURCE_DIR = 'data'
# TARGET_DIR = 'data_augmented'
# TARGET_IMAGES_PER_CLASS = 2000

# # --- Augmentation Pipeline ---
# # NOTE: We do NOT use ToTensor() or Normalize() here because we are saving as image files.
# augmentation_pipeline = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=45),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
# ])

# def create_augmented_dataset():
#     """
#     Creates a new, balanced dataset by augmenting images from a source directory.
#     """
#     print(f"Starting offline augmentation...")
#     print(f"Source directory: '{SOURCE_DIR}'")
#     print(f"Target directory: '{TARGET_DIR}'")
#     print(f"Target images per class: {TARGET_IMAGES_PER_CLASS}")
    
#     # Get class names from the source directory
#     class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
#     if not class_names:
#         print(f"Error: No class subdirectories found in '{SOURCE_DIR}'.")
#         return

#     # Create the target directory if it doesn't exist
#     os.makedirs(TARGET_DIR, exist_ok=True)

#     for class_name in class_names:
#         print(f"\nProcessing class: {class_name}")
        
#         source_class_path = os.path.join(SOURCE_DIR, class_name)
#         target_class_path = os.path.join(TARGET_DIR, class_name)
#         os.makedirs(target_class_path, exist_ok=True)
        
#         # Get list of original images
#         original_images = glob.glob(os.path.join(source_class_path, '*.jpg'))
        
#         if not original_images:
#             print(f"Warning: No images found for class '{class_name}'. Skipping.")
#             continue
            
#         # --- 1. First, copy all original images to the new directory ---
#         image_count = 0
#         for img_path in original_images:
#             img_name = os.path.basename(img_path)
#             target_path = os.path.join(target_class_path, img_name)
#             Image.open(img_path).convert("RGB").save(target_path)
#             image_count += 1
            
#         # --- 2. Now, generate augmented images until the target is reached ---
#         num_to_generate = TARGET_IMAGES_PER_CLASS - image_count
        
#         if num_to_generate <= 0:
#             print(f"Class '{class_name}' already has {image_count} images. No new images needed.")
#             continue
            
#         print(f"Copying {image_count} original images...")
#         print(f"Generating {num_to_generate} new augmented images...")
        
#         progress_bar = tqdm(range(num_to_generate), desc=f"Augmenting {class_name}")
        
#         for i in progress_bar:
#             # Pick a random original image to augment
#             random_original_path = random.choice(original_images)
#             original_image = Image.open(random_original_path).convert("RGB")
            
#             # Apply the augmentation pipeline
#             augmented_image = augmentation_pipeline(original_image)
            
#             # Save the new image with a unique name
#             new_image_name = f"aug_{i+1}_{os.path.basename(random_original_path)}"
#             augmented_image.save(os.path.join(target_class_path, new_image_name))
            
#     print("\nOffline augmentation complete!")
#     print(f"New dataset is ready in '{TARGET_DIR}'.")

# if __name__ == "__main__":
#     create_augmented_dataset()




# create_augmented_dataset.py (CORRECTED VERSION)

import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

# --- Configuration ---
SOURCE_DIR = 'data'
TARGET_DIR = 'data_augmented'
TARGET_IMAGES_PER_CLASS = 2000

# --- Augmentation Pipeline ---
# NOTE: We do NOT use ToTensor() or Normalize() here because we are saving as image files.
augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
])

def create_augmented_dataset():
    """
    Creates a new, balanced dataset by augmenting images from a source directory.
    This version uses a more robust method to handle special characters in folder names.
    """
    print(f"Starting offline augmentation...")
    print(f"Source directory: '{SOURCE_DIR}'")
    print(f"Target directory: '{TARGET_DIR}'")
    print(f"Target images per class: {TARGET_IMAGES_PER_CLASS}")
    
    # Get class names from the source directory
    class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not class_names:
        print(f"Error: No class subdirectories found in '{SOURCE_DIR}'.")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)

    for class_name in class_names:
        print("-" * 50)
        print(f"Processing class: {class_name}")
        
        source_class_path = os.path.join(SOURCE_DIR, class_name)
        target_class_path = os.path.join(TARGET_DIR, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # --- ROBUST FILE FINDING ---
        # Using os.listdir() is safer than glob() for paths with special characters like '[]'
        original_filenames = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        original_image_paths = [os.path.join(source_class_path, f) for f in original_filenames]
        
        print(f"Found {len(original_image_paths)} original images for class '{class_name}'.")

        if not original_image_paths:
            print(f"Warning: No images found. Skipping.")
            continue
            
        # --- 1. First, copy all original images to the new directory ---
        print(f"Copying {len(original_image_paths)} original images...")
        for img_path in original_image_paths:
            img_name = os.path.basename(img_path)
            target_path = os.path.join(target_class_path, img_name)
            Image.open(img_path).convert("RGB").save(target_path)
        
        # --- 2. Now, generate augmented images until the target is reached ---
        current_image_count = len(original_image_paths)
        num_to_generate = TARGET_IMAGES_PER_CLASS - current_image_count
        
        if num_to_generate <= 0:
            print(f"Class '{class_name}' already has {current_image_count} images. No new images needed.")
            continue
            
        print(f"Generating {num_to_generate} new augmented images...")
        
        progress_bar = tqdm(range(num_to_generate), desc=f"Augmenting {class_name}")
        
        for i in progress_bar:
            # Pick a random original image to augment
            random_original_path = random.choice(original_image_paths)
            original_image = Image.open(random_original_path).convert("RGB")
            
            # Apply the augmentation pipeline
            augmented_image = augmentation_pipeline(original_image)
            
            # Save the new image with a unique name
            new_image_name = f"aug_{i+1}_{os.path.basename(random_original_path)}"
            augmented_image.save(os.path.join(target_class_path, new_image_name))
            
    print("\n" + "="*50)
    print("Offline augmentation complete!")
    print(f"New, balanced dataset is ready in '{TARGET_DIR}'.")
    print("="*50)

if __name__ == "__main__":
    create_augmented_dataset()