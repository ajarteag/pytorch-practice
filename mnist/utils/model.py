import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import TypeAlias, List

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

class SimpleMLP(nn.Module):
    """Simple Three Linear Layer Fully Connected Architecture for MNIST."""
    def __init__(self, hidden_layer_size: int = 64, num_classes: int = 10):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(28 * 28, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_classes)
    
    def forward(self, x: InputTensor) -> OutputTensor:
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# define ResNet50 NN Architecture
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block: Block, layers: List[int], image_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet_layers = nn.Sequential(
            self._make_layer(block, layers[0], out_channels=64, stride=1),
            self._make_layer(block, layers[1], out_channels=128, stride=2),
            self._make_layer(block, layers[2], out_channels=256, stride=2),
            self._make_layer(block, layers[3], out_channels=512, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resnet_layers(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4 , kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))  # 256 -> 64, 64*4 (256) again
        
        return nn.Sequential(*layers)
    
class ResNet50(ResNet):
    def __init__(self, img_channels: int = 1, num_classes: int = 10):
        super().__init__(
            block=Block, 
            layers=[3, 4, 6, 3], 
            image_channels=img_channels, 
            num_classes=num_classes
        )
    
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