import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models import NeuralNetMLP
from training import custom_train, train_torch_model
from utils import compute_mse_and_acc, plot_training_curves


def main(num_epochs, num_hidden, minibatch_size, learning_rate):
    SEED = 10
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load and preprocess MNIST data
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.values.astype(np.float32)
    y = mnist.target.values.astype(np.int64)

    # Scale pixels to [-1, 1]
    X = ((X / X[0].max()) - 0.5) * 2

    # Split into train, validation, test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=10000, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=5000, random_state=SEED, stratify=y_temp)

    num_features = X_train.shape[1]
    num_classes = 10

    # --------- Custom Model ---------
    print("Training custom numpy NN model...")
    custom_model = NeuralNetMLP(num_features, num_hidden, num_classes)
    c_loss, c_train_acc, c_valid_acc = custom_train(
        custom_model, X_train, y_train, X_valid, y_valid,
        num_epochs, minibatch_size, learning_rate
    )

    # --------- PyTorch Model ---------
    print("\nTraining PyTorch model...")
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.int64)
    X_valid_t = torch.tensor(X_valid)
    y_valid_t = torch.tensor(y_valid, dtype=torch.int64)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    valid_dataset = torch.utils.data.TensorDataset(X_valid_t, y_valid_t)

    model = nn.Sequential(
        nn.Linear(num_features, num_hidden),
        nn.Sigmoid(),
        nn.Linear(num_hidden, num_classes),
        nn.Sigmoid()
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    t_loss, t_train_acc, t_valid_acc = train_torch_model(
        model, train_dataset, valid_dataset,
        num_epochs, minibatch_size, loss_fn, optimizer
    )

    # --------- Final Test Accuracy ---------
    print("\nEvaluating on test set...")
    _, c_test_acc = compute_mse_and_acc(custom_model, X_test, y_test, custom=True, minibatch_size=minibatch_size)
    _, t_test_acc = compute_mse_and_acc(model, X_test, y_test, custom=False, minibatch_size=minibatch_size)

    print(f"Custom NN Test Accuracy: {c_test_acc * 100:.2f}%")
    print(f"PyTorch NN Test Accuracy: {t_test_acc * 100:.2f}%")

    # --------- Plot Results ---------
    plot_training_curves(c_loss, c_train_acc, c_valid_acc, t_loss, t_train_acc, t_valid_acc)


if __name__ == "__main__":
    # Hyperparameters: feel free to tweak these
    NUM_EPOCHS = 50
    NUM_HIDDEN = 50
    MINIBATCH_SIZE = 100
    LEARNING_RATE = 0.1

    main(
        num_epochs=NUM_EPOCHS,
        num_hidden=NUM_HIDDEN,
        minibatch_size=MINIBATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
