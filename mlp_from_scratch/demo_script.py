import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models import NeuralNetMLP, TwoLayersNeuralNetMLP
from training import (
    custom_train_mse, custom_train_ce,
    train_torch_model
)
from utils import (
    compute_mse_and_acc, compute_ce_and_acc,
    plot_training_curves
)


def main(num_epochs, num_h1, num_h2, minibatch_size, learning_rate,
         use_two_layer=False, use_cross_entropy=False):
    
    SEED = 10
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load and preprocess MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.values.astype(np.float32)
    y = mnist.target.values.astype(np.int64)

    # Normalize to [-1, 1]
    X = ((X / X[0].max()) - 0.5) * 2

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=10000, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=5000, random_state=SEED, stratify=y_temp)

    num_features = X_train.shape[1]
    num_classes = 10

    # ========== Custom NumPy Model ==========
    print(f"Training custom NumPy model ({'2-layer' if use_two_layer else '1-layer'})...")
    if use_two_layer:
        custom_model = TwoLayersNeuralNetMLP(num_features, num_h1, num_h2, num_classes)
        c_loss, c_train_acc, c_valid_acc = custom_train_ce(
            custom_model, X_train, y_train, X_valid, y_valid,
            num_epochs, minibatch_size, learning_rate)
    else:
        custom_model = NeuralNetMLP(num_features, num_h1, num_classes)
        c_loss, c_train_acc, c_valid_acc = custom_train_mse(
            custom_model, X_train, y_train, X_valid, y_valid,
            num_epochs, minibatch_size, learning_rate)

    # ========== PyTorch Model ==========
    print(f"\nTraining PyTorch model ({'2-layer' if use_two_layer else '1-layer'})...")
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.int64)
    X_valid_t = torch.tensor(X_valid)
    y_valid_t = torch.tensor(y_valid, dtype=torch.int64)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    valid_dataset = torch.utils.data.TensorDataset(X_valid_t, y_valid_t)

    if use_two_layer:
        model = nn.Sequential(
            nn.Linear(num_features, num_h1),
            nn.Sigmoid(),
            nn.Linear(num_h1, num_h2),
            nn.Sigmoid(),
            nn.Linear(num_h2, num_classes)
        )
    else:
        model = nn.Sequential(
            nn.Linear(num_features, num_h1),
            nn.Sigmoid(),
            nn.Linear(num_h1, num_classes),
            nn.Sigmoid() if not use_cross_entropy else nn.Identity()
        )

    if use_cross_entropy:
        loss_fn = nn.CrossEntropyLoss()
        loss_type = 'ce'
    else:
        loss_fn = nn.MSELoss()
        loss_type = 'mse'

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    t_loss, t_train_acc, t_valid_acc = train_torch_model(
        model, train_dataset, valid_dataset,
        num_epochs, minibatch_size, loss_fn, optimizer,
        loss_type=loss_type
    )

    # ========== Final Test Accuracy ==========
    print("\nEvaluating on test set...")
    if use_cross_entropy:
        c_test_loss, c_test_acc = compute_ce_and_acc(custom_model, X_test, y_test, custom=True, minibatch_size=minibatch_size)
        t_test_loss, t_test_acc = compute_ce_and_acc(model, X_test, y_test, custom=False, minibatch_size=minibatch_size)
    else:
        c_test_loss, c_test_acc = compute_mse_and_acc(custom_model, X_test, y_test, custom=True, minibatch_size=minibatch_size)
        t_test_loss, t_test_acc = compute_mse_and_acc(model, X_test, y_test, custom=False, minibatch_size=minibatch_size)

    print(f"Custom Model Test Accuracy: {c_test_acc * 100:.2f}%")
    print(f"PyTorch Model Test Accuracy: {t_test_acc * 100:.2f}%")

    # ========== Plot ==========
    plot_training_curves(
        c_loss, c_train_acc, c_valid_acc,
        t_loss, t_train_acc, t_valid_acc
    )


if __name__ == "__main__":
    # --- Hyperparameters ---
    NUM_EPOCHS = 50
    NUM_H1 = 50
    NUM_H2 = 30
    MINIBATCH_SIZE = 100
    LEARNING_RATE = 0.1

    # === Toggle Config Here ===
    USE_TWO_LAYER = True
    USE_CROSS_ENTROPY = True  # Must be True for 2-layer

    main(
        num_epochs=NUM_EPOCHS,
        num_h1=NUM_H1,
        num_h2=NUM_H2,
        minibatch_size=MINIBATCH_SIZE,
        learning_rate=LEARNING_RATE,
        use_two_layer=USE_TWO_LAYER,
        use_cross_entropy=USE_CROSS_ENTROPY
    )
