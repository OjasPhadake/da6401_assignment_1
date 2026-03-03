"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, x):
        f = self.forward(x)
        return f * (1 - f)

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)

class Tanh:
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x)**2

class Softmax:
    def forward(self, x):
        # Shift for numerical stability
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    # Backward is usually handled within the Cross-Entropy loss derivative