import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import TypeAlias

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

InputTensor: TypeAlias = torch.FloatTensor  # Input Shape: ([B] C H W)
OutputTensor: TypeAlias = torch.FloatTensor  # Output Shape: ([B] K)

# --- Model Architecture ---

class NeuralNet(nn.Module):
    """Simple Three Linear Layer Fully Connected Architecture for MNIST."""
    def __init__(self, hidden_layer_size: int = 64, num_classes: int = 10):
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
            nn.Linear(hidden_layer_size, num_classes)
        )
    
    def forward(self, x: InputTensor) -> OutputTensor:
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
# --- Reusable Evaluation Function ---

def evaluate_accuracy(model: nn.Module, data_loader: DataLoader, device: str) -> float:
    """Calculates the classification accuracy on a given DataLoader."""
    total_samples = 0
    total_correct = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicts = torch.max(outputs, dim=1)
            
            total_samples += len(labels)
            total_correct += torch.sum(predicts == labels).item()
            
    accuracy = 100.0 * total_correct / total_samples
    return accuracy