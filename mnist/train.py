'''
Author: Alejandro Arteaga <ajarteag@usc.edu>

This is script is meant to practice 
    - downloading a dataset from torchvision {MNIST or FashionMNIST}
    - train a simple DNN image classification model
    - test the accuracy of the trained model
    - compare the accuracy of the trained model to existing models
    - save the trained model weights dict to a .pt file
'''

import torch
from torch import nn
import argparse
from pathlib import Path
from typing import NamedTuple, List
from utils.model import SimpleMLP, ResNet50, evaluate_accuracy
from utils.data import load_data

# --- Configuration ---
class HParams(NamedTuple):
    dataset_name: str
    model_arch: str
    train_ratio: float
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    seed: int
    hidden_layer_sizes: List[int]

DEFAULT_HPARAMS = HParams(
    dataset_name='mnist', # This default will be overridden by the mandatory CL argument
    model_arch='simplemlp',
    train_ratio=0.8,
    batch_size=128,
    num_workers=0,
    epochs=10,
    lr=0.01,
    seed=817,
    hidden_layer_sizes=[16, 32, 64, 128]
)

DATASET_INFO = {
    'mnist': {
        'description': 'MNIST (Handwritten Digits)', 
        'output_classes': 10,
        'image_shape': '(1, 28, 28) (Greyscale, Flattened to 784)',
        'labels': '10 (Digits 0-9)'
    },
    'fashionmnist': {
        'description': 'FashionMNIST (Zalando Clothing Items)',
        'output_classes': 10,
        'image_shape': '(1, 28, 28) (Greyscale, Flattened to 784)',
        'labels': '10 (Clothing Categories)'
    }
}

MODEL_ARCHITECTURES = {
    'simplemlp': SimpleMLP,
    'resnet50': ResNet50
}

# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments, setting mandatory and optional values."""
    parser = argparse.ArgumentParser(description="Train a DNN on MNIST or FashionMNIST.")
    
    # Mandatory Arguments
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True,
        choices=['mnist', 'fashionmnist'],
        help='Name of the dataset to use: "mnist" or "fashionmnist".'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        required=True,
        choices=list(MODEL_ARCHITECTURES.keys()),
        help='Model architecture to use: "simplemlp" or "resnet50".'
    )
    
    # Optional Arguments (with default values from DEFAULT_HPARAMS)
    parser.add_argument(
        '--train_ratio', 
        type=float, 
        default=DEFAULT_HPARAMS.train_ratio,
        help='Ratio of the full training set to use for actual training (0.0 to 1.0).'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=DEFAULT_HPARAMS.batch_size,
        help='Input batch size for training.'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=DEFAULT_HPARAMS.num_workers,
        help='Number of subprocesses to use for data loading. Set to 0 for single-process.'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=DEFAULT_HPARAMS.epochs,
        help='Number of training epochs per hidden layer size.'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=DEFAULT_HPARAMS.lr,
        help='Learning rate for the Adam optimizer.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=DEFAULT_HPARAMS.seed,
        help='Random seed for reproducibility.'
    )
    
    args = parser.parse_args()
    
    # Consolidate parsed args with fixed/default constants into the final HParams object
    final_hparams = HParams(
        dataset_name=args.dataset_name,
        model_arch=args.model_arch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        hidden_layer_sizes=DEFAULT_HPARAMS.hidden_layer_sizes # Using the fixed constant
    )
    return final_hparams


def print_run_summary(hparams: HParams, device):
    """Prints a consolidated summary of the dataset, config, and model."""
    info = DATASET_INFO.get(hparams.dataset_name.lower(), None)
    if info is None:
        raise ValueError(f"Unknown dataset: {hparams.dataset_name}")
    
    print("=====================================================")
    print(f"           {hparams.dataset_name.upper()} DNN TRAINING SUMMARY 🚀             ")
    print("=====================================================")
    
    print("\n[DATASET AND LABELS]")
    print(f"Dataset:            {info['description']}")
    print(f"Image Shape:        {info['image_shape']}")
    print(f"Output Classes:     {info['labels']}")
    # Use hparams.TRAIN_RATIO
    print(f"Data Split:         {hparams.train_ratio*100:.0f}% Train / {100-hparams.train_ratio*100:.0f}% Validation")
    
    print("\n[TRAINING CONFIGURATION]")
    print(f"Device:             {device}")
    # Use hparams.LR
    print(f"Optimizer:          Adam (Learning Rate: {hparams.lr})")
    print(f"Loss Function:      CrossEntropyLoss (for 10 classes)")
    # Use hparams.EPOCHS, hparams.BATCH_SIZE, hparams.NUM_WORKERS
    print(f"Epochs per HLS:     {hparams.epochs}")
    print(f"Batch Size:         {hparams.batch_size}")
    print(f"Data Workers:       {hparams.num_workers} (for faster data loading)")

    print("\n[MODEL ARCHITECTURE (NeuralNet)]")
    if hparams.model_arch == 'simplemlp':
        print("Type:               Fully Connected (MLP)")
        print(f"Input Layer:        784 features (28*28)")
        print(f"Hidden Layers:      2 layers with ReLU activation")
        print(f"Output Layer:       10 logits (one for each class)")
        print(f"HLS Search Space:   {hparams.hidden_layer_sizes}")
    elif hparams.model_arch == 'resnet50':
        print("Type:               ResNet50 (Convolutional)")
        print("Structure:          Bottleneck blocks, 4 stages")
        print(f"Input Channels:     1 (Greyscale)")
        print(f"Output Layer:       10 logits (one for each class)")
    print("=====================================================\n")


def train_and_validate(hls, train_loader, val_loader, device, loss_fn, hparams: HParams):
    """Handles the training and validation for a single model configuration."""
    model_class = MODEL_ARCHITECTURES[hparams.model_arch]
    
    # Instantiate the correct model based on model_arch
    if hparams.model_arch == 'simplemlp':
        model = model_class(hidden_layer_size=hls)
    elif hparams.model_arch == 'resnet50':
        # ResNet50 is fixed and ignores hls, but we iterate over hls to match the outer loop structure
        model = model_class(img_channels=1, num_classes=10)
    else:
        # Should not happen due to argparse choices, but good for safety
        raise ValueError(f"Unsupported model architecture: {hparams.model_arch}")
    
    model.to(device=device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr)
    
    model.train()
    
    for epoch in range(hparams.epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
        log_hls = f'hls={hls:3d}, ' if hparams.model_arch == 'simplemlp' else ''
        print(f'{log_hls}epoch={epoch+1:2d}, loss={loss.item():.3f}')

    val_accuracy = evaluate_accuracy(model, val_loader, device)
    print(f'(hls={hls}) validation accuracy = {val_accuracy:.2f}%')
    return model, val_accuracy


def load_and_test_existing_model(save_path: Path, best_hls: int, test_loader, device, model_arch: str):
    """Loads the state dict of an existing model and evaluates its accuracy."""
    print(f"Found existing model at {save_path}. Evaluating...")
    
    model_class = MODEL_ARCHITECTURES[model_arch]
    
    # Instantiate the correct model class
    if model_arch == 'simplemlp':
        existing_model = model_class(hidden_layer_size=best_hls)
    elif model_arch == 'resnet50':
        existing_model = model_class(img_channels=1, num_classes=10)
    existing_model.to(device)

    try:
        existing_model.load_state_dict(
            torch.load(
                str(save_path), 
                map_location=device,
                weights_only=True
            )
        )
    except Exception as e:
        print(f"Error loading existing model weights: {e}")
        return 0.0

    existing_accuracy = evaluate_accuracy(existing_model, test_loader, device)
    print(f'Existing model test accuracy: {existing_accuracy:.2f}%')
    
    return existing_accuracy

def main():
    hparams = parse_args() # Get configuration from command line

    # 1. Setup Device and Data Loaders
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print_run_summary(hparams=hparams, device=device)
    
    generator = torch.Generator().manual_seed(hparams.seed)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = load_data(
        dataset_name=hparams.dataset_name,
        train_ratio=hparams.train_ratio,
        batch_size=hparams.batch_size,
        generator=generator,
        num_workers=hparams.num_workers
    )

    # 2. Hyperparameter Search
    max_accuracy = 0.0
    best_model = None
    best_hls = None

    # Handle search space: ResNet50 is a fixed model, SimpleMLP iterates through HLS
    search_space = hparams.hidden_layer_sizes if hparams.model_arch == 'simplemlp' else [0] # Use [0] for a single ResNet50 run

    print("\n--- Starting Hyperparameter Search ---")
    for hls in search_space:
        model, accuracy = train_and_validate(
            hls, 
            train_loader, 
            val_loader, 
            device, 
            loss_fn, 
            hparams
        )
        
        if accuracy > max_accuracy:
            print('New best model found.')
            max_accuracy = accuracy
            best_model = model
            best_hls = hls

    print("\n--- Final Evaluation and Checkpoint ---")
    model_detail = f'_hls{best_hls}' if hparams.model_arch == 'simplemlp' else ''
    save_path = Path(f'./models/{hparams.dataset_name}_{hparams.model_arch}{model_detail}.pt')
    
    current_test_accuracy = evaluate_accuracy(best_model, test_loader, device)
    print(f'Current run best model ({hparams.model_arch}{model_detail}) test accuracy = {current_test_accuracy:.2f}%')

    # Check if a model already exists at the save_path
    if save_path.exists():
        existing_test_accuracy = load_and_test_existing_model(
            save_path=save_path, 
            best_hls=best_hls, 
            test_loader=test_loader, 
            device=device,
            model_arch=hparams.model_arch
        )
        
        if current_test_accuracy > existing_test_accuracy:
            torch.save(best_model.state_dict(), str(save_path)) 
            print(f'✅ Model UPDATED: New test accuracy ({current_test_accuracy:.2f}%) > Existing ({existing_test_accuracy:.2f}%).')
        else:
            print(f'❌ Model NOT SAVED: Existing test accuracy ({existing_test_accuracy:.2f}%) is higher or equal.')
            
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model.state_dict(), str(save_path))
        print(f'⭐ Model SAVED: No existing model found. Saved to {save_path}')

if __name__ == "__main__":
    main()