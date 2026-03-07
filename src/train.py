"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import sys
import os
# Fix the path first!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import argparse
import wandb
import numpy as np
import json
import os
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    # Dataset and training basics
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    
    # Architecture and Optimization
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop'], default='sgd')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=str, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'], default='relu')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier'], default='xavier')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy')

    # Mandatory W&B and Save arguments [cite: 427]
    # parser.add_argument('-w_p', '--wandb_project', type=str, required=True, help='W&B Project ID')
    parser.add_argument('-w_p', '--wandb_project', required=False, default="autograder_test")
    parser.add_argument('--model_save_path', type=str, default='src/best_model.npy')
    
    return parser.parse_args()

def log_data_exploration(x_train, y_train, dataset_name):
    """
    Logs 5 sample images per class to a W&B Table for MNIST or Fashion-MNIST.
    """
    if dataset_name == 'fashion_mnist':
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    else: # Default to MNIST
        class_names = ["0 - Zero", "1 - One", "2 - Two", "3 - Three", "4 - Four", 
                       "5 - Five", "6 - Six", "7 - Seven", "8 - Eight", "9 - Nine"]

    columns = ["image", "label", "class_name"]
    table = wandb.Table(columns=columns)

    for class_id in range(10):
        # Get first 5 indices for this specific class
        idx = np.where(y_train == class_id)[0][:5]
        for i in idx:
            # Reshape from flattened 784 back to 28x28 for the UI
            img = x_train[i].reshape(28, 28)
            table.add_data(wandb.Image(img), class_id, class_names[class_id])
    
    # Log once to create the 'Artifact' in your dashboard
    wandb.log({"Data Exploration Table": table})
    
def main():
    args = parse_arguments()
    if isinstance(args.hidden_size, list) and isinstance(args.hidden_size[0], str):
        full_str = "".join(args.hidden_size)
        if "[" in full_str:
            import ast
            args.hidden_size = ast.literal_eval(full_str)
        else:
            # If they were just space separated numbers like 128 64
            args.hidden_size = [int(x) for x in args.hidden_size]
            
    wandb.init(project=args.wandb_project, config=vars(args))
    
    (x_train, y_train), (x_val, y_val) = load_data(args.dataset)
    if isinstance(args.hidden_size, str):
    # Convert "[128, 128, 128]" to [128, 128, 128]
        import ast
        args.hidden_size = ast.literal_eval(args.hidden_size)
        
    model = NeuralNetwork(args) # Pass the full args object
    log_data_exploration(x_train, y_train, args.dataset)

    best_f1 = -1.0
    # Training logic calling model.train()
    model.train(x_train, y_train, args.epochs, args.batch_size)
    
    val_accuracy = model.evaluate(x_val, y_val)
    y_pred = model.pred(x_val)
    
    # Log to wandb summary so the sweep can see the final result
    wandb.log({"val_accuracy": val_accuracy})
    wandb.log({
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,                # Use None if passing discrete preds
            y_true=y_val, 
            preds=y_pred,
            class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        )
    })
    
    metrics = {
        "val_accuracy": accuracy_score(y_val, y_pred),
        "val_precision": precision_score(y_val, y_pred, average='macro'),
        "val_recall": recall_score(y_val, y_pred, average='macro'),
        "val_f1": f1_score(y_val, y_pred, average='macro')
    }

    wandb.log(metrics)
    
    wandb.run.summary["val_accuracy"] = val_accuracy
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    best_weights = model.get_weights()
    np.save(args.model_save_path, best_weights)
    
    with open("src/best_config.json", "w") as f:
        json.dump(vars(args), f)

    print(f"Model saved to {args.model_save_path}")

if __name__ == '__main__':
    main()