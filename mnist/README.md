# Readme: DNN Classification Trainer 🚀

The train.py script provides a command-line interface to train a simple Deep Neural Network (DNN) for image classification on either the **MNIST** or **FashionMNIST** datasets. It includes a hyperparameter search and implements intelligent checkpointing to save the model only if it achieves a higher test accuracy than any previously saved version.

## 💻 System Requirements

This project is built using PyTorch and is optimized for the **Metal Performance Shaders (MPS)** backend on Apple Silicon for GPU acceleration.

- **Operating System**: macOS (M-series, e.g., M1, M2, M3)
- **Package Manager**: Mamba/Conda (recommended)
- **Python**: Version 3.10+
- **PyTorch**: Latest stable version supporting the MPS backend (e.g., **torch >= 2.4.1**).

Recommended Environment Setup (Mamba/Conda)

For the best performance on your M-series Mac, use a native `osx-arm64` environment:

```bash
# 1. Create the environment
mamba create -n dnn_train python=3.10

# 2. Activate the environment
mamba activate dnn_train

# 3. Install PyTorch and dependencies (this command pulls the native arm64 build with MPS support)
mamba install pytorch torchvision torchaudio -c pytorch
```

## ▶️ How to Run the Script

The script uses `argparse` and requires the `--dataset_name` argument. All other training parameters are optional and use sensible defaults.

### 1. Mandatory Dataset Training

You must specify either `mnist` or `fashionmnist`. All other hyperparameters will use their default values (e.g., `epochs=10`, `batch_size=128`, `lr=0.01`).

```bash
# Train the DNN on the MNIST dataset using default settings
python train.py --dataset_name mnist

# Train the DNN on the FashionMNIST dataset using default settings
python train.py --dataset_name fashionmnist
```

### 2. Customizing Hyperparameters

You can override any of the optional hyperparameters for more detailed experiments:

```bash
# Example: Train FashionMNIST with a larger batch size, fewer epochs, and a lower learning rate.
python train.py \
    --dataset_name fashionmnist \
    --batch_size 256 \
    --epochs 5 \
    --lr 0.005
```

## 💾 Checkpointing and Saving

The script automatically manages model checkpoints in the `./models` directory.

The model is named based on the dataset and the best performing Hidden Layer Size (HLS) found during the search:

- **Save Path Format**: `./models/{dataset_name}_classification_mlp_hls{best_hls}.pt`

### Intelligent Checkpointing Logic

- After finding the best model on the validation set, it calculates its accuracy on the independent test set.
- If a file already exists at the save_path, it loads the existing model and evaluates its test accuracy.
- The script saves the new model only if its test accuracy is strictly higher than the existing model's test accuracy. Otherwise, the existing (superior) model is preserved.
