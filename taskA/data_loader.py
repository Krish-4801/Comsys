# src/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from . import config

def create_dataloader(data_dir, batch_size, shuffle=True):
    """
    Creates a DataLoader for a given dataset directory.

    Args:
        data_dir (str): Path to the dataset directory (e.g., '.../train').
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the dataset.
        list: The class names found in the directory.
    """
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=os.cpu_count() // 2, # Use half of the available CPU cores
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return loader, dataset.classes