# src/config.py
# src/config.py

import torch

# --- Project Paths ---
# CHANGE THIS LINE to point to the new augmented dataset
DATA_DIR = 'data_augmented' 
OUTPUT_DIR = 'outputs'

# --- Data Parameters ---
# ... (the rest of the file remains the same)

# --- Data Parameters ---
IMG_SIZE = 224
# Class names must match the folder names in DATA_DIR
CLASS_NAMES = ['Benign', '[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B']
NUM_CLASSES = len(CLASS_NAMES)

# --- Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 25 # Starting value, can be tuned
LEARNING_RATE = 1e-4
NUM_FOLDS = 5 # For stratified k-fold cross-validation