# train.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from model import build_sequential_model

# Load and preprocess data
fashion = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plotting function for predictions
def plot_predictions(model, X_test, y_test, row=1, col=8, figsize=(15, 3)):
    prob = model.predict(X_test, verbose=0)
    pred_labels = prob.argmax(axis=1)
    correct = np.where(pred_labels == y_test)[0]
    wrong = np.where(pred_labels != y_test)[0]

    # Wrong predictions
    print("\nWRONG CLASSIFICATIONS")
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, idx in zip(range(len(axs)), wrong):
        axs[i].imshow(X_test[idx], cmap='Greys')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f"Pred: {class_names[pred_labels[idx]]}")
        axs[i].set_title(f"GT: {class_names[y_test[idx]]}")
    plt.tight_layout()
    plt.show()

    # Correct predictions
    print("\nCORRECT CLASSIFICATIONS")
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, idx in zip(range(len(axs)), correct):
        axs[i].imshow(X_test[idx], cmap='Greys')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f"Pred: {class_names[pred_labels[idx]]}")
        axs[i].set_title(f"GT: {class_names[y_test[idx]]}")
    plt.tight_layout()
    plt.show()

# Build model
input_shape = X_train.shape[1:]
model = build_sequential_model(input_shape)

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.SGD(),
    metrics=[keras.metrics.sparse_categorical_accuracy]
)

# Evaluate before training
_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy Before Training: {acc * 100:.2f}%")

# Train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    verbose=0
)

# Evaluate after training
_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy After Training: {acc * 100:.2f}%")

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Training and Validation Accuracy/Loss")
plt.show()

# Show predictions
plot_predictions(model, X_test, y_test)
