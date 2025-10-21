import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from typing import Tuple

# Global/static data configurations
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

def get_data_transforms():
    """Defines the standard transforms for MNIST data."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

def load_mnist_data(train_ratio: float, 
                    data_path: str = "./data",
                    batch_size: int = 32, 
                    generator: torch.Generator = None, 
                    shuffle_train: bool = True,
                    num_workers: int = 0) -> Tuple[MNIST, MNIST, MNIST]:
    """Loads MNIST, splits it, and returns DataLoaders."""
    transform = get_data_transforms()
    full_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_path, train=False, transform=transform)

    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset=full_dataset, 
        lengths=[train_size, val_size], 
        generator=generator
    )

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