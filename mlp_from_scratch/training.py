import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import (minibatch_generator, compute_mse_and_acc, 
                   custom_collate_fn)

def custom_train(model, X_train, y_train, X_valid, y_valid, 
                 num_epochs, minibatch_size, learning_rate=0.1):
    """
    Train a custom neural network model (numpy-based) using mini-batch gradient descent.

    Args:
        model: Custom neural network instance with forward and backward methods.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_valid (np.ndarray): Validation features.
        y_valid (np.ndarray): Validation labels.
        num_epochs (int): Number of training epochs.
        minibatch_size (int): Mini-batch size.
        learning_rate (float): Learning rate for gradient descent.

    Returns:
        Tuple of lists: (epoch_loss, epoch_train_acc, epoch_valid_acc)
    """
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_batch, y_batch in minibatch_gen:
            # Forward pass
            a_h, a_out = model.forward(X_batch)

            # Backward pass: compute gradients
            d_w_out, d_b_out, d_w_h, d_b_h = model.backward(X_batch, a_h, a_out, y_batch)

            # Update weights and biases using gradient descent
            model.weight_h -= learning_rate * d_w_h
            model.bias_h -= learning_rate * d_b_h
            model.weight_out -= learning_rate * d_w_out
            model.bias_out -= learning_rate * d_b_out

        # Compute metrics after epoch
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train, custom=True, minibatch_size=minibatch_size)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid, custom=True, minibatch_size=minibatch_size)

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train MSE: {train_mse:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Valid Acc: {valid_acc*100:.2f}%")

    return epoch_loss, epoch_train_acc, epoch_valid_acc


def train_torch_model(model, train_dataset, valid_dataset, num_epochs, minibatch_size, loss_fn, optimizer):
    """
    Train a PyTorch model using DataLoader and standard PyTorch workflow.

    Args:
        model: PyTorch nn.Module model.
        train_dataset: torch.utils.data.Dataset for training.
        valid_dataset: torch.utils.data.Dataset for validation.
        num_epochs (int): Number of epochs.
        minibatch_size (int): Batch size.
        loss_fn: PyTorch loss function (e.g., nn.MSELoss()).
        optimizer: PyTorch optimizer (e.g., torch.optim.SGD).

    Returns:
        Tuple of lists: (epoch_loss, epoch_train_acc, epoch_valid_acc)
    """
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True,
                              collate_fn=custom_collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=minibatch_size, shuffle=False,
                              collate_fn=custom_collate_fn)

    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Convert y_batch to one-hot for MSE loss
            y_one_hot = torch.zeros(y_batch.size(0), y_pred.size(1))
            y_one_hot.scatter_(1, y_batch.unsqueeze(1), 1)

            loss = loss_fn(y_pred, y_one_hot)
            loss.backward()
            optimizer.step()

        # Evaluate training performance
        model.eval()
        with torch.no_grad():
            train_features = train_dataset.tensors[0].numpy()
            train_labels = train_dataset.tensors[1].numpy()
            valid_features = valid_dataset.tensors[0].numpy()
            valid_labels = valid_dataset.tensors[1].numpy()

            train_mse, train_acc = compute_mse_and_acc(model, train_features, train_labels, custom=False, minibatch_size=minibatch_size)
            valid_mse, valid_acc = compute_mse_and_acc(model, valid_features, valid_labels, custom=False, minibatch_size=minibatch_size)

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train MSE: {train_mse:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Valid Acc: {valid_acc*100:.2f}%")

    return epoch_loss, epoch_train_acc, epoch_valid_acc
