# src/data_utils/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class BloodCellDataset(Dataset):
    """
    Custom PyTorch Dataset for loading blood cell images.
    """
    def __init__(self, image_paths, labels, class_to_idx, transforms=None):
        """
        Args:
            image_paths (list): List of full paths to each image.
            labels (list): List of integer labels corresponding to the image_paths.
            class_to_idx (dict): Mapping from class name to integer label.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transforms = transforms
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the integer label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image and convert to RGB
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy sample or handle appropriately
            return torch.randn(3, 224, 224), -1 

        # Apply transformations
        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)