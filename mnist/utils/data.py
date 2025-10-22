import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Literal

# Global/static data configurations
# Mean and STD for standard MNIST dataset (0-255 images normalized to 0-1)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
# Mean and STD for FashionMNIST dataset (0-255 images normalized to 0-1)
FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)

def tensor_normalized_transform(mean: Tuple[float], std: Tuple[float]):
    """Defines the standard transforms for MNIST data."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def load_data(dataset_name: Literal["mnist", "fashionmnist"],
              train_ratio: float, 
              data_path: str = "./data",
              batch_size: int = 32, 
              generator: torch.Generator = None, 
              shuffle_train: bool = True,
              num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads MNIST/FashionMNIST, splits it, and returns DataLoaders."""
    # Select Dataset Class and Normalization Stats
    if dataset_name.lower() == "mnist":
        DatasetClass = datasets.MNIST
        mean, std = MNIST_MEAN, MNIST_STD
    elif dataset_name.lower() == "fashionmnist":
        DatasetClass = datasets.FashionMNIST
        mean, std = FASHIONMNIST_MEAN, FASHIONMNIST_STD
    else:
        raise ValueError("Invalid dataset_name. Choose 'mnist' or 'fashionmnist'.")

    # Define Transform
    transform = tensor_normalized_transform(mean, std)

    # Load Datasets
    full_dataset = DatasetClass(root=data_path, train=True, download=True, transform=transform)
    test_dataset = DatasetClass(root=data_path, train=False, download=True, transform=transform)

    # Split Training Data
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset=full_dataset, 
        lengths=[train_size, val_size], 
        generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader