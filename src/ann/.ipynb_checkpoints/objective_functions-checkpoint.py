"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def cross_entropy(y_pred, y_true):
    """
    Assumes y_pred are probabilities (Softmax output) 
    and y_true are integer labels[cite: 92].
    """
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-15)
    loss = np.sum(log_likelihood) / m
    return loss

def mse(y_pred, y_true_onehot):
    return np.mean(np.square(y_pred - y_true_onehot))

def get_loss_grad(y_pred, y_true, loss_type):
    m = y_true.shape[0]
    if loss_type == 'cross_entropy':
        # Combined Gradient of Softmax + Cross-Entropy
        grad = y_pred.copy()
        grad[range(m), y_true] -= 1
        return grad # This is dZ for the last layer
    # Add MSE logic if needed