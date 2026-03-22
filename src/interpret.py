# src/interpret.py

# src/interpret.py (FULLY REVISED IMPORTS)

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from PIL import Image
import os  # <--- THIS WAS THE MISSING LINE

# --- Import project modules ---
from src import config
from src.models import cnn_models
from src.data_utils import augmentations

class GradCAM:
    """
    Grad-CAM implementation to visualize model attention.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Forward pass to get activations and predictions
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target = output[0][class_idx]
        target.backward()
        
        # Get activations and gradients
        activations = self.activations.detach().cpu()
        gradients = self.gradients.detach().cpu()
        
        # Pool gradients across channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight the channels by the gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.numpy(), class_idx

def overlay_heatmap(image, heatmap, colormap=cv2.COLORMAP_JET):
    """Overlays the heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    overlayed_image = heatmap * 0.4 + image * 0.6
    overlayed_image = np.clip(overlayed_image, 0, 255).astype(np.uint8)
    return overlayed_image

def get_target_layer(model, model_name):
    if model_name.startswith('resnet'):
        return model.layer4[-1]
    elif model_name.startswith('efficientnet'):
        return model.features[-1]
    elif model_name.startswith('densenet'):
        return model.features.norm5
    else:
        raise ValueError(f"Target layer not defined for model {model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a model and image.")
    parser.add_argument('--model-name', type=str, required=True, choices=['resnet50', 'efficientnet_b0', 'densenet121'], help='Model architecture.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the output Grad-CAM image.')
    args = parser.parse_args()

    # --- 1. Load Model ---
    print(f"Loading model: {args.model_name} from {args.model_path}")
    model_func = getattr(cnn_models, f"get_{args.model_name}")
    model = model_func(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)

    # --- FIX: Unfreeze all model parameters to allow gradients to flow ---
    for param in model.parameters():
        param.requires_grad = True

    model.eval()

    # --- 2. Load and Preprocess Image ---
    print(f"Loading image from: {args.image_path}")
    original_image = cv2.imread(args.image_path)
    pil_image = Image.open(args.image_path).convert("RGB")
    
    preprocess = augmentations.get_val_transforms()
    input_tensor = preprocess(pil_image).unsqueeze(0).to(config.DEVICE)

    # --- 3. Generate Grad-CAM ---
    target_layer = get_target_layer(model, args.model_name)
    grad_cam = GradCAM(model, target_layer)
    
    heatmap, predicted_idx = grad_cam.generate_heatmap(input_tensor)
    
    # Get class names
    idx_to_class = {i: name for i, name in enumerate(config.CLASS_NAMES)}
    predicted_class = idx_to_class[predicted_idx]
    print(f"Model prediction: {predicted_class}")
    
    # --- 4. Overlay and Save ---
    grad_cam_image = overlay_heatmap(original_image, heatmap)
    
    if args.save_path is None:
        base_name = os.path.basename(args.image_path)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(config.OUTPUT_DIR, 'grad_cam_visuals', f'{name}_gradcam.png')
    else:
        save_path = args.save_path
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, grad_cam_image)
    print(f"Grad-CAM visualization saved to: {save_path}")