"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

class NeuralLayer:
    def __init__(self, n_in, n_out, activation_type, init_method='xavier'):
        # Weight Initialization [cite: 51, 102]
        if init_method == 'xavier':
            limit = np.sqrt(6 / (n_in + n_out))
            self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        else: # Random [cite: 51]
            self.W = np.random.randn(n_in, n_out) * 0.01
            
        self.b = np.zeros((1, n_out))
        
        # Gradient storage for autograder verification [cite: 53, 168]
        self.grad_W = None
        self.grad_b = None
        
        # Cache for backward pass
        self.input_cache = None
        self.z_cache = None
        self.activation_fn = None

    def forward(self, A_prev):
        """ Linear step: Z = WA + b """
        self.input_cache = A_prev
        self.z_cache = np.dot(A_prev, self.W) + self.b
        return self.z_cache

    def backward(self, dZ):
        # Only apply the activation derivative if it exists and we aren't at the output layer
        # if hasattr(self.activation_fn, 'backward'):
        #     dZ = dZ * self.activation_fn.backward(self.z_cache)
        
        batch_size = self.input_cache.shape[0]
        self.grad_W = np.dot(self.input_cache.T, dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size
        
        return np.dot(dZ, self.W.T)