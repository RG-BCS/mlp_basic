import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils import (
    minibatch_generator, int_to_onehot,
    compute_mse_and_acc, compute_ce_and_acc,
    custom_collate_fn
)

# === 1. CUSTOM TRAINING (NumPy-based)

def custom_train_mse(model, X_train, y_train, X_valid, y_valid, 
                     num_epochs, minibatch_size, learning_rate=0.1):
    """
    Train custom (NumPy) single-layer model with MSE loss.
    """
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        for X_batch, y_batch in minibatch_generator(X_train, y_train, minibatch_size):
            a_h, a_out = model.forward(X_batch)
            d_w_out, d_b_out, d_w_h, d_b_h = model.backward(X_batch, a_h, a_out, y_batch)

            model.weight_h -= learning_rate * d_w_h
            model.bias_h -= learning_rate * d_b_h
            model.weight_out -= learning_rate * d_w_out
            model.bias_out -= learning_rate * d_b_out

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train, custom=True, minibatch_size=minibatch_size)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid, custom=True, minibatch_size=minibatch_size)

        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Custom MSE] Epoch {epoch+1}/{num_epochs} | Train MSE: {train_mse:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | Valid Acc: {valid_acc*100:.2f}%")

    return epoch_loss, epoch_train_acc, epoch_valid_acc


def custom_train_ce(model, X_train, y_train, X_valid, y_valid, 
                    num_epochs, minibatch_size, learning_rate=0.1):
    """
    Train custom (NumPy) two-layer model with Cross-Entropy loss.
    """
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        for X_batch, y_batch in minibatch_generator(X_train, y_train, minibatch_size):
            a_h1, a_h2, a_out = model.forward(X_batch)
            grads = model.backward(X_batch, a_h1, a_h2, a_out, y_batch)

            d_w_out, d_b_out, d_w_h2, d_b_h2, d_w_h1, d_b_h1 = grads

            model.weight_out -= learning_rate * d_w_out
            model.bias_out -= learning_rate * d_b_out
            model.weight_h_2 -= learning_rate * d_w_h2
            model.bias_h_2 -= learning_rate * d_b_h2
            model.weight_h_1 -= learning_rate * d_w_h1
            model.bias_h_1 -= learning_rate * d_b_h1

        train_loss, train_acc = compute_ce_and_acc(model, X_train, y_train, custom=True, minibatch_size=minibatch_size)
        valid_loss, valid_acc = compute_ce_and_acc(model, X_valid, y_valid, custom=True, minibatch_size=minibatch_size)

        epoch_loss.append(train_loss)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Custom CE] Epoch {epoch+1}/{num_epochs} | Train CE: {train_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | Valid Acc: {valid_acc*100:.2f}%")

    return epoch_loss, epoch_train_acc, epoch_valid_acc


# === 2. TORCH TRAINING

def train_torch_model(model, train_dataset, valid_dataset, num_epochs, minibatch_size,
                      loss_fn, optimizer, loss_type='mse'):
    """
    Train PyTorch model using DataLoader.
    Set `loss_type` to 'mse' or 'ce' to match the model output.
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
            logits = model(x_batch)

            if loss_type == 'mse':
                # Convert to one-hot
                y_one_hot = torch.zeros(y_batch.size(0), logits.size(1))
                y_one_hot.scatter_(1, y_batch.unsqueeze(1), 1)
                loss = loss_fn(logits, y_one_hot)
            else:  # CrossEntropy
                loss = loss_fn(logits, y_batch)

            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            X_train_tensor, y_train_tensor = train_dataset.tensors
            X_valid_tensor, y_valid_tensor = valid_dataset.tensors

            train_np_X = X_train_tensor.numpy()
            train_np_y = y_train_tensor.numpy()
            valid_np_X = X_valid_tensor.numpy()
            valid_np_y = y_valid_tensor.numpy()

            if loss_type == 'mse':
                train_loss, train_acc = compute_mse_and_acc(model, train_np_X, train_np_y, custom=False, minibatch_size=minibatch_size)
                valid_loss, valid_acc = compute_mse_and_acc(model, valid_np_X, valid_np_y, custom=False, minibatch_size=minibatch_size)
            else:
                from utils import compute_ce_and_acc
                train_loss, train_acc = compute_ce_and_acc(model, train_np_X, train_np_y, custom=False, minibatch_size=minibatch_size)
                valid_loss, valid_acc = compute_ce_and_acc(model, val_
