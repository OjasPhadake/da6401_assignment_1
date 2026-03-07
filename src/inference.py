import sys
import os
# Fix the path first!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_data

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
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'], default='relu')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier'], default='xavier')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy')

    # Mandatory W&B and Save arguments [cite: 427]
    # parser.add_argument('-w_p', '--wandb_project', type=str, required=True, help='W&B Project ID')
    parser.add_argument('-w_p', '--wandb_project', required=False, default="autograder_test")
    # parser.add_argument('--model_path', type=str, default='src/best_model.npy', help='Path to saved weights')    
    default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.npy')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to saved weights')
    
    return parser.parse_args()

def load_model(model_path):
    """ Load trained model from disk as per instructions [cite: 437, 439] """
    data = np.load(model_path, allow_pickle=True).item()
    return data

def main():
    args = parse_arguments()
    _, (x_test, y_test) = load_data(args.dataset) # Load test split
    
    # Initialize model and load weights [cite: 441, 443]
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
      
    # During inference or testing
    logits = model.forward(x_test) # Test with a single sample
    preds = np.argmax(logits, axis=1) # Logits work for argmax
    
    # Calculate Metrics [cite: 422]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average='macro'),
        "recall": recall_score(y_test, preds, average='macro'),
        "f1": f1_score(y_test, preds, average='macro')
    }
    
    print("Inference Metrics:", metrics)
    return metrics

if __name__ == '__main__':
    main()