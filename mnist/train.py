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
from utils.model import NeuralNet, evaluate_accuracy
from utils.data import load_mnist_data

# --- Configuration ---
# Hyperparameters
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
NUM_WORKERS = 0
EPOCHS = 10
LR = 0.01
SEED = 817

# Model Search Space
HIDDEN_LAYER_SIZES = [16, 32, 64, 128]

def print_run_summary(device):
    """Prints a consolidated summary of the dataset, config, and model."""
    print("=====================================================")
    print("           MNIST DNN TRAINING SUMMARY 🚀             ")
    print("=====================================================")
    
    print("\n[DATASET AND LABELS]")
    print(f"Dataset:            MNIST (Handwritten Digits)")
    print(f"Image Shape:        (1, 28, 28) (Greyscale, Flattened to 784)")
    print(f"Output Classes:     10 (Digits 0-9)")
    print(f"Data Split:         {TRAIN_RATIO*100:.0f}% Train / {100-TRAIN_RATIO*100:.0f}% Validation")
    
    print("\n[TRAINING CONFIGURATION]")
    print(f"Device:             {device}")
    print(f"Optimizer:          Adam (Learning Rate: {LR})")
    print(f"Loss Function:      CrossEntropyLoss (for 10 classes)")
    print(f"Epochs per HLS:     {EPOCHS}")
    print(f"Batch Size:         {BATCH_SIZE}")
    print(f"Data Workers:       {NUM_WORKERS} (for faster data loading)")

    print("\n[MODEL ARCHITECTURE (NeuralNet)]")
    print("Type:               Fully Connected (MLP)")
    print(f"Input Layer:        784 features (28*28)")
    print(f"Hidden Layers:      2 layers with ReLU activation")
    print(f"Output Layer:       10 logits (one for each class)")
    print(f"HLS Search Space:   {HIDDEN_LAYER_SIZES}")
    print("=====================================================\n")

def train_and_validate(hls, train_loader, val_loader, device, loss_fn):
    """Handles the training and validation for a single model configuration."""
    model = NeuralNet(hidden_layer_size=hls)
    model.to(device=device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    
    model.train()
    
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
        print(f'hls={hls:3d}, epoch={epoch+1:2d}, loss={loss.item():.3f}')

    val_accuracy = evaluate_accuracy(model, val_loader, device)
    print(f'(hls={hls}) validation accuracy = {val_accuracy:.2f}%')
    return model, val_accuracy


def main():
    # 1. Setup Device and Data Loaders
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print_run_summary(device=device)
    
    generator = torch.Generator().manual_seed(SEED)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = load_mnist_data(
        train_ratio=TRAIN_RATIO,
        batch_size=BATCH_SIZE,
        generator=generator,
        num_workers=NUM_WORKERS
    )

    # 2. Hyperparameter Search
    max_accuracy = 0.0
    best_model = None
    best_hls = None

    print("\n--- Starting Hyperparameter Search ---")
    for hls in HIDDEN_LAYER_SIZES:
        model, accuracy = train_and_validate(hls, train_loader, val_loader, device, loss_fn)
        
        if accuracy > max_accuracy:
            print('New best model found.')
            max_accuracy = accuracy
            best_model = model
            best_hls = hls

    # 3. Final Test and Saving
    print("\n--- Final Evaluation ---")
    test_accuracy = evaluate_accuracy(best_model, test_loader, device)
    print(f'Best model (HLS={best_hls}) test accuracy = {test_accuracy:.2f}%')
    
    # Save the best model's state dict
    save_path = './models/mlp_mnist_classification.pt'
    torch.save(best_model.state_dict(), save_path)
    print(f'Model weights saved to {save_path}')

if __name__ == "__main__":
    main()