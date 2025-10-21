'''
Author: Alejandro Arteaga <ajarteag@usc.edu>

This is script is meant to practice 
    - downloading a dataset from torchvision
    - train a simple DNN image classification model
    - test the accuracy of the trained model
    - save the trained model weights dict to a .pt file
    - compare the accuracy of the trained model to existing models
'''

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from typing import List, Tuple, TypeAlias

'''
MNIST data consistant of greyscale images of handrawn digits with shape (C, H, W)
    - C =  1: number of color channels 
    - H = 28: height of image in pixels
    - W = 28: width of image in pixels

Data trained in batches of size B
    - B = 32: batch_size

Output is a K-dimensional vector for K image classes, in this case digits 0-9
    - K = 10: number of classes
'''

InputTensor: TypeAlias = torch.FloatTensor  # Shape: (B) C H W
OutputTensor: TypeAlias = torch.FloatTensor  # Shape: (B) K

def load_mnist_data(train_ratio: float, 
                    data_path: str = "./data",
                    batch_size: int = 32, 
                    generator: torch.Generator = None, 
                    shuffle_train: bool = True,
                    num_workers: int = 0) -> Tuple[MNIST, MNIST, MNIST]:
    mean = (0.1307,)
    std = (0.3081,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
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


'''
Simple Three Linear Layer Fully Connected Architecture
'''
class NeuralNet(nn.Module):
    def __init__(self, hidden_layer_size: int = 64):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, hidden_layer_size),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_layer_size, 10)
        )
    
    def forward(self, x: InputTensor):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def main():
    # Define paramters for run (future: change to commandline arguments)
    train_ratio = 0.8
    data_path = './data'
    batch_size = 128
    generator = torch.Generator().manual_seed(817)
    shuffle_train = True
    num_workers = 4
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'using device = {device}')
    epochs = 10
    lr = 0.01
    loss_fn = nn.CrossEntropyLoss()

    # Load & split data into training, validation, and test datasets
    train_loader, val_loader, test_loader = load_mnist_data(
        train_ratio=train_ratio,
        data_path=data_path,
        batch_size=batch_size,
        generator=generator,
        shuffle_train=shuffle_train,
        num_workers=num_workers
    )

    # Future: Use Cross-Validation to tune the model hyper-parameters
    hidden_layer_sizes = [16, 32, 64, 128]
    max_accuracy = 0
    best_model = None
    best_hls = None
    for hidden_layer_size in hidden_layer_sizes:
        # Instantiate the model with initialized random weights (future: use scheduler + momentum)
        model = NeuralNet(hidden_layer_size=hidden_layer_size)
        model.to(device=device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        print(f'training model with hidden_layer_size (hls) = {hidden_layer_size}')
        model.train()
        for epoch in range(epochs):
            # Train the model and save model with best validation accuracy
            for inputs, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'epoch = {epoch+1}, loss = {loss*len(labels):.3f}')
        total_samples = 0
        total_correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicts = torch.max(outputs, dim=1)
                total_samples += len(labels)
                total_correct += torch.sum(predicts == labels).item()
        accuracy = 100 * total_correct / total_samples
        print(f'(hls={hidden_layer_size}) validation accuracy = {accuracy:.2f}')
        if accuracy > max_accuracy:
            print('new best model updated')
            max_accuracy = accuracy
            best_model = model
            best_hls = hidden_layer_size

    # Report model accuracy & save model to model.pt file
    total_samples = 0
    total_correct = 0
    best_model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicts = torch.max(outputs, dim=1)
            total_samples += len(labels)
            total_correct += torch.sum(predicts == labels).item()
    test_accuracy = 100 * total_correct / total_samples
    print(f'best model (hls={best_hls}) test accuracy = {test_accuracy:.2f}')
    torch.save(best_model.state_dict, './models/mlp_model.pt')

if __name__ == "__main__":
    main()