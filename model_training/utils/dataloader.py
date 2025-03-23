from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

def get_split_loaders(data_dir, train_transform, val_transform, batch_size=32, split_ratio=0.8):
    # Load full dataset
    full_dataset = ImageFolder(root=data_dir)

    # Calculate lengths
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split dataset
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms manually to subsets
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

## This is just random spliting, we need to create a CustomDataloader for straitified splitting. However, since dataset size is large, the class imbalance is small and manageable.