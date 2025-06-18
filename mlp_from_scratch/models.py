import numpy as np
from utils import int_to_onehot, sigmoid, softmax

import torch
import torch.nn as nn

class NeuralNetMLP:
    """
    A simple MLP neural network from scratch using NumPy,
    with one hidden layer and sigmoid activations.
    """
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        rng = np.random.RandomState(seed=random_seed)
        self.num_classes = num_classes
        
        # Initialize weights and biases for hidden layer
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # Initialize weights and biases for output layer
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
    
    def sigmoid(self, z):
        # Sigmoid activation function
        return 1. / (1. + np.exp(-z))
    
    def forward(self, x, training=True):
        """
        Forward pass through the network.
        x: input batch, shape (batch_size, num_features)
        Returns:
          - if training: (hidden_layer_activations, output_probabilities)
          - if eval: output_probabilities only
        """
        # Hidden layer linear transformation + activation
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = self.sigmoid(z_h)
        
        # Output layer linear transformation + activation
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = self.sigmoid(z_out)
        
        if not training:
            return a_out
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        """
        Backpropagation to compute gradients.
        x: input batch
        a_h: hidden layer activations from forward pass
        a_out: output activations from forward pass
        y: true labels (integers)
        Returns gradients for all weights and biases.
        """
        # Convert labels to one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)
        
        # Gradient of loss wrt output activations (MSE loss)
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        
        # Derivative of sigmoid at output layer
        d_a_out__d_z_out = a_out * (1. - a_out)
        
        # Delta for output layer
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        
        # Gradients for output weights and biases
        d_loss__dw_out = np.dot(delta_out.T, a_h)  # shape: (num_classes, num_hidden)
        d_loss__db_out = np.sum(delta_out, axis=0) # shape: (num_classes,)
        
        # Backpropagate delta to hidden layer
        d_loss__a_h = np.dot(delta_out, self.weight_out)  # shape: (batch_size, num_hidden)
        
        # Derivative of sigmoid at hidden layer
        d_a_h__d_z_h = a_h * (1. - a_h)
        
        # Gradients for hidden weights and biases
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, x)  # shape: (num_hidden, num_features)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)  # shape: (num_hidden,)
        
        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h


class TorchMLP(nn.Module):
    """
    A simple MLP built with PyTorch, with one hidden layer and sigmoid activations,
    for comparison with the custom vanilla MLP.
    """
    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()
        self.hidden = nn.Linear(num_features, num_hidden)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(num_hidden, num_classes)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the PyTorch model.
        Input x shape: (batch_size, num_features)
        Returns output probabilities of shape (batch_size, num_classes).
        """
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.output_activation(x)
        return x

# --- Existing classes NeuralNetMLP and TorchMLP above ---

class Two_Layers_NeuralNetMLP:
    """
    Two-hidden-layer MLP from scratch using NumPy.
    Hidden layers use sigmoid activation, output uses softmax.
    Backpropagation implements gradient computation.
    """
    def __init__(self, num_features, num_h1, num_h2, num_classes, random_seed=123):
        rng = np.random.RandomState(seed=random_seed)
        self.num_classes = num_classes
        
        # Layer 1 weights and biases
        self.weight_h_1 = rng.normal(loc=0.0, scale=0.1, size=(num_h1, num_features))
        self.bias_h_1 = np.zeros(num_h1)
        
        # Layer 2 weights and biases
        self.weight_h_2 = rng.normal(loc=0.0, scale=0.1, size=(num_h2, num_h1))
        self.bias_h_2 = np.zeros(num_h2)
        
        # Output layer weights and biases
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_h2))
        self.bias_out = np.zeros(num_classes)
    
    def forward(self, x, training=True):
        # Forward pass: input → hidden1 → hidden2 → output
        
        # Hidden layer 1
        z_h1 = np.dot(x, self.weight_h_1.T) + self.bias_h_1
        a_h1 = sigmoid(z_h1)
        
        # Hidden layer 2
        z_h2 = np.dot(a_h1, self.weight_h_2.T) + self.bias_h_2
        a_h2 = sigmoid(z_h2)
        
        # Output layer (softmax)
        z_out = np.dot(a_h2, self.weight_out.T) + self.bias_out
        a_out = softmax(z_out)
        
        if not training:
            return a_out
        return a_h1, a_h2, a_out

    def backward(self, x, a_h1, a_h2, a_out, y):
        # Convert labels to one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)
        
        # Output layer error (softmax + cross-entropy derivative)
        delta_out = (a_out - y_onehot) / y.shape[0]
        
        # Gradients output layer
        d_loss_dw_out = np.dot(delta_out.T, a_h2)
        d_loss_db_out = np.sum(delta_out, axis=0)
        
        # Backprop to hidden layer 2
        d_loss_a_h2 = np.dot(delta_out, self.weight_out)
        d_a_h2_d_z_h2 = a_h2 * (1. - a_h2)
        delta_h2 = d_loss_a_h2 * d_a_h2_d_z_h2
        
        d_loss_dw_h2 = np.dot(delta_h2.T, a_h1)
        d_loss_db_h2 = np.sum(delta_h2, axis=0)
        
        # Backprop to hidden layer 1
        d_loss_a_h1 = np.dot(delta_h2, self.weight_h_2)
        d_a_h1_d_z_h1 = a_h1 * (1. - a_h1)
        delta_h1 = d_loss_a_h1 * d_a_h1_d_z_h1
        
        d_loss_dw_h1 = np.dot(delta_h1.T, x)
        d_loss_db_h1 = np.sum(delta_h1, axis=0)
        
        return d_loss_dw_out, d_loss_db_out, d_loss_dw_h2, d_loss_db_h2, d_loss_dw_h1, d_loss_db_h1


class TorchMLP2Layers(nn.Module):
    """
    A 2-hidden-layer MLP in PyTorch using sigmoid activations and raw logits.
    Suitable for CrossEntropyLoss (no softmax at the output).
    """
    def __init__(self, num_features, num_h1, num_h2, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, num_h1),
            nn.Sigmoid(),
            nn.Linear(num_h1, num_h2),
            nn.Sigmoid(),
            nn.Linear(num_h2, num_classes)  # Raw logits (no softmax)
        )

    def forward(self, x):
        return self.model(x)
