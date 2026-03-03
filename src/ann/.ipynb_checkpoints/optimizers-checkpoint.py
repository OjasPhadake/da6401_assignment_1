"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.001, weight_decay=0.0):
        self.lr = learning_rate
        self.wd = weight_decay

    def apply_weight_decay(self, layer):
        if self.wd > 0:
            layer.grad_W += self.wd * layer.W
            layer.grad_b += self.wd * layer.b

class SGD(Optimizer):
    def step(self, layers):
        for layer in layers:
            self.apply_weight_decay(layer)
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.gamma = momentum
        self.v_W = {}
        self.v_b = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)
            
            self.apply_weight_decay(layer)
            self.v_W[i] = self.gamma * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.gamma * self.v_b[i] + self.lr * layer.grad_b
            
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]

class NAG(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.gamma = momentum
        self.v_W = {}
        self.v_b = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)
            
            self.apply_weight_decay(layer)
            v_W_prev = self.v_W[i]
            v_b_prev = self.v_b[i]
            
            self.v_W[i] = self.gamma * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.gamma * self.v_b[i] + self.lr * layer.grad_b
            
            # Look-ahead update logic
            layer.W -= (self.gamma * self.v_W[i] + self.lr * layer.grad_W)
            layer.b -= (self.gamma * self.v_b[i] + self.lr * layer.grad_b)

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.eps = epsilon
        self.v_W = {}
        self.v_b = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)
            
            self.apply_weight_decay(layer)
            self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * (layer.grad_W**2)
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (layer.grad_b**2)
            
            layer.W -= (self.lr / (np.sqrt(self.v_W[i]) + self.eps)) * layer.grad_W
            layer.b -= (self.lr / (np.sqrt(self.v_b[i]) + self.eps)) * layer.grad_b