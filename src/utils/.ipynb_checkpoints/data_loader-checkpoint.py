"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
The load_data function
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # This hides most of those warnings
from tensorflow.keras.datasets import mnist, fashion_mnist

def load_data(dataset_name):
    # dataset loading
    if dataset_name == 'mnist':
        (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset. Choose between 'mnist' or 'fashion_mnist'.")


    # preprocessing of the dataset
    x_train_full = x_train_full.reshape(x_train_full.shape[0], 784).astype('float32')/255.0
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')/255.0

    num_samples = x_train_full.shape[0]
    indices = np.random.permutation(num_samples)
    
    # Split: 50,000 for training, 10,000 for validation (standard for MNIST)
    split_idx = int(num_samples * 0.8333) # Approximately 50,000
    
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples (for inference): {x_test.shape[0]}")

    return (x_train, y_train), (x_val, y_val)
