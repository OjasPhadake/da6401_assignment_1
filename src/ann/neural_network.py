"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
import wandb
from src.ann.neural_layer import NeuralLayer
from src.ann.activations import Sigmoid, ReLU, Tanh, Softmax
from src.ann.objective_functions import cross_entropy, mse, get_loss_grad
from src.ann.optimizers import SGD, Momentum, NAG, RMSProp

print("DEBUG: neural_network.py successfully loaded")

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize the neural network based on CLI arguments. [cite: 35]
        """
        self.args = cli_args
        self.layers = []
        
        # Determine activation class
        activation_map = {'sigmoid': Sigmoid, 'relu': ReLU, 'tanh': Tanh}
        self.hidden_activation = activation_map[self.args.activation]

        # 1. Build Architecture [cite: 56]
        current_dim = 784 # Flattened 28x28 images
        
        for size in self.args.hidden_size:
            layer = NeuralLayer(current_dim, size, self.args.activation, self.args.weight_init)
            layer.activation_fn = self.hidden_activation()
            self.layers.append(layer)
            current_dim = size
            
        # 2. Output Layer (10 classes)
        output_layer = NeuralLayer(current_dim, 10, 'softmax', self.args.weight_init)
        output_layer.activation_fn = Softmax()
        self.layers.append(output_layer)

        # 3. Initialize Optimizer [cite: 45]
        opt_map = {
            'sgd': SGD, 'momentum': Momentum, 'nag': NAG, 
            'rmsprop': RMSProp
        }
        self.optimizer = opt_map[self.args.optimizer](
            learning_rate=self.args.learning_rate, 
            weight_decay=self.args.weight_decay
        )
    
    def forward(self, X):
        """
        Forward propagation. Returns raw logits for the output layer. [cite: 60, 176]
        """
        A = X
        for i, layer in enumerate(self.layers):
            Z = layer.forward(A)
            # Final layer returns raw logits (linear combination) 
            if i == len(self.layers) - 1:
                return Z
            A = layer.activation_fn.forward(Z)
            layer.A_cache = A 
        return A
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute and return gradients. 
        """
        # Note: y_pred here refers to probabilities calculated for the loss internally
        # but the autograder expects backward to receive ground truth and predictions.
        dZ = get_loss_grad(y_pred, y_true, self.args.loss)
        
        grads = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dA_prev = layer.backward(dZ)
            
            # Record gradients for return 
            grads.append((layer.grad_W, layer.grad_b))
            
            if i > 0:
                prev_layer = self.layers[i-1]
                # Chain Rule: dZ_prev = dA_prev * f'(Z_prev)
                dZ = dA_prev * prev_layer.activation_fn.backward(prev_layer.z_cache)
        
        # Return gradients from last layer to first 
        return grads

    def get_weights(self):
        """
        Returns a dictionary of current weights and biases for all layers. 
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'W{i}'] = layer.W
            weights[f'b{i}'] = layer.b
        return weights

    def set_weights(self, weights):
        """
        Sets the weights and biases for all layers from a dictionary. [cite: 180, 190]
        """
        for i, layer in enumerate(self.layers):
            layer.W = weights[f'W{i}']
            layer.b = weights[f'b{i}']
    
    def update_weights(self):
        self.optimizer.step(self.layers)
    
    def train(self, X_train, y_train, epochs, batch_size):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_s, y_s = X_train[indices], y_train[indices]
            
            total_loss = 0
            for j in range(0, num_samples, batch_size):
                X_batch = X_s[j : j + batch_size]
                y_batch = y_s[j : j + batch_size]
                
                # Forward to get logits
                logits = self.forward(X_batch)
                
                # Internal softmax for loss computation
                probs = Softmax().forward(logits)
                
                if self.args.loss == 'cross_entropy':
                    batch_loss = cross_entropy(probs, y_batch)
                else:
                    y_true_oh = np.eye(10)[y_batch]
                    batch_loss = mse(probs, y_true_oh)
                
                total_loss += batch_loss
                self.backward(y_batch, probs)
                layer_0_grad_W = self.layers[0].grad_W
                
                grad_logs = {
                    f"grad_neuron_{i}": np.mean(np.abs(layer_0_grad_W[:, i]))
                    for i in range(min(5, layer_0_grad_W.shape[1]))
                }
                wandb.log(grad_logs)
                grad_norm = np.linalg.norm(self.layers[0].grad_W)
                wandb.log({"grad_norm_first_layer": grad_norm})
                
                self.update_weights()
            
            avg_loss = total_loss / (num_samples // batch_size)
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def evaluate(self, X, y):
        logits = self.forward(X)
        probs = Softmax().forward(logits)
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == y)

    def pred(self, X):
        logits = self.forward(X)
        probs = Softmax().forward(logits) # Probabilities
        predictions = np.argmax(probs, axis=1) # Predicted classes
        return predictions