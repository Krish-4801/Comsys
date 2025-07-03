# src/model.py
import torch.nn as nn
from torchvision import models
from . import config

def get_model(pretrained=config.PRETRAINED, num_classes=config.NUM_CLASSES):
    """
    Loads a pretrained ResNet-18 model and replaces the final
    fully connected layer to match the number of classes.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The configured model.
    """
    model = models.resnet18(pretrained=pretrained)
    
    # Get the number of input features for the classifier
    in_features = model.fc.in_features
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(in_features, num_classes)
    
    return model