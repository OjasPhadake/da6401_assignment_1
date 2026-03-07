# DA6401 Assignment 1: Multi-Layer Perceptron

This repository contains the implementation of a custom Multi-Layer Perceptron (MLP) for image classification (MNIST and Fashion-MNIST). The project includes modules for neural layers, activation functions, loss calculation, and various optimization algorithms.

## Project Structure

* `src/ann/`:
* `neural_network.py`: Orchestrates training and inference loops.
* `neural_layer.py`: Handles linear operations and weight initialization.
* `activations.py`: Implements activation functions (ReLU, Sigmoid, Tanh, Softmax).
* `objective_functions.py`: Implements Cross-Entropy and MSE loss.
* `optimizers.py`: Implements optimization algorithms (SGD, Momentum, RMSProp).


## Usage

To run the model, use the `train.py` script (ensure you have `wandb` configured):

```bash
python train.py --activation relu --optimizer adam --hidden_size 256 128 --weight_init xavier

```

## Experimental Results

The comprehensive analysis, including gradient plots, hyperparameter tuning, and theory question responses, can be found in the Weights & Biases report below:

* **GitHub Repository:** [https://github.com/OjasPhadake/da6401_assignment_1](https://github.com/OjasPhadake/da6401_assignment_1)
* **W&B Report:** [https://api.wandb.ai/links/ch22b007-indian-institute-of-technology-madras/5zz82vsr](https://api.wandb.ai/links/ch22b007-indian-institute-of-technology-madras/5zz82vsr)
