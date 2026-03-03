# """
# Inference Script
# Evaluate trained models on test sets
# """

# import argparse

# def parse_arguments():
#     """
#     Parse command-line arguments for inference.
    
#     TODO: Implement argparse with:
#     - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
#     - dataset: Dataset to evaluate on
#     - batch_size: Batch size for inference
#     - hidden_layers: List of hidden layer sizes
#     - num_neurons: Number of neurons in hidden layers
#     - activation: Activation function ('relu', 'sigmoid', 'tanh')
#     """
#     parser = argparse.ArgumentParser(description='Run inference on test set')
    
#     return parser.parse_args()


# def load_model(model_path):
#     """
#     Load trained model from disk.
#     """
#     pass


# def evaluate_model(model, X_test, y_test): 
#     """
#     Evaluate model on test data.
        
#     TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
#     """
#     pass


# def main():
#     """
#     Main inference function.

#     TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
#     """
#     args = parse_arguments()
    
#     print("Evaluation complete!")


# if __name__ == '__main__':
#     main()


# """
# Inference Script
# Evaluate trained models on test sets
# """

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_data

def parse_arguments():
    # CLI for inference must match train.py 
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('-w_p', '--wandb_project', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='src/best_model.npy')
    # Include same architecture arguments as train.py to initialize the model
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    # ... include other mandatory args from train.py ...
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
    
    # Forward pass gets raw logits 
    logits = model.forward(x_test)
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