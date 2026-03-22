# src/models/cnn_models.py

import torch
import torch.nn as nn
from torchvision import models
from src import config

def _set_requires_grad(model, requires_grad):
    """
    A helper function to freeze or unfreeze all model parameters.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_resnet50(num_classes=config.NUM_CLASSES, pretrained=True, freeze_layers=True):
    """
    Loads a pre-trained ResNet-50 model and replaces its final classification layer.

    Args:
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to use pre-trained weights from ImageNet.
        freeze_layers (bool): If True, freezes the weights of the convolutional layers.

    Returns:
        A PyTorch model (ResNet-50).
    """
    # Load the pre-trained model
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze the convolutional layers if specified
    if freeze_layers:
        _set_requires_grad(model, False)

    # Replace the final fully connected layer (the 'head')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    # Ensure the new layers are trainable
    # This is redundant if freeze_layers is True and we built a new head,
    # but it is good practice to be explicit.
    _set_requires_grad(model.fc, True)

    return model

def get_efficientnet_b0(num_classes=config.NUM_CLASSES, pretrained=True, freeze_layers=True):
    """
    Loads a pre-trained EfficientNet-B0 model and replaces its final classification layer.

    Args:
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to use pre-trained weights from ImageNet.
        freeze_layers (bool): If True, freezes the weights of the feature extraction layers.

    Returns:
        A PyTorch model (EfficientNet-B0).
    """
    # Load the pre-trained model
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze the feature extraction layers if specified
    if freeze_layers:
        _set_requires_grad(model.features, False)

    # Replace the final classifier layer
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )

    # Ensure the new layers are trainable
    _set_requires_grad(model.classifier, True)

    return model

# --- You can add more models here following the same pattern ---
# Example: DenseNet121
def get_densenet121(num_classes=config.NUM_CLASSES, pretrained=True, freeze_layers=True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    if freeze_layers:
        _set_requires_grad(model.features, False)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    _set_requires_grad(model.classifier, True)

    return model

if __name__ == '__main__':
    # A quick test to see if the models load correctly
    print("--- Testing Model Architectures ---")

    device = config.DEVICE
    print(f"Using device: {device}")

    # Test ResNet-50
    resnet = get_resnet50().to(device)
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    output = resnet(dummy_input)
    print(f"ResNet-50 loaded successfully. Output shape: {output.shape}") # Expected: [2, 4]

    # Test EfficientNet-B0
    efficientnet = get_efficientnet_b0().to(device)
    output = efficientnet(dummy_input)
    print(f"EfficientNet-B0 loaded successfully. Output shape: {output.shape}") # Expected: [2, 4]

    # Test DenseNet-121
    densenet = get_densenet121().to(device)
    output = densenet(dummy_input)
    print(f"DenseNet-121 loaded successfully. Output shape: {output.shape}") # Expected: [2, 4]

    print("---------------------------------")