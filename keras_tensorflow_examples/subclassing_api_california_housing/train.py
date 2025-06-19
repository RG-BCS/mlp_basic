# train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import WideAndDeepModel
from tensorflow import keras

# Load and preprocess dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

X_train_A, X_train_B = X_train_scaled[:, :5], X_train_scaled[:, 2:]
X_valid_A, X_valid_B = X_valid_scaled[:, :5], X_valid_scaled[:, 2:]
X_test_A, X_test_B = X_test_scaled[:, :5], X_test_scaled[:, 2:]

# Instantiate and compile model
model = WideAndDeepModel()
model.compile(
    loss=['mse', 'mse'],
    loss_weights=[0.9, 0.1],
    optimizer='sgd'
)

# Evaluate before training
y_test_pair = (y_test.reshape(-1, 1), y_test.reshape(-1, 1))
loss_before = model.custom_evaluate_((X_test_A, X_test_B), y_test_pair, verbose=0)
print(f"Subclassing API Test MSE Before Training: {loss_before:.2f}")

# Train model
history = model.fit(
    [X_train_A, X_train_B],
    [y_train, y_train],
    epochs=20,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]),
    verbose=0
)

# Evaluate after training
loss_after = model.custom_evaluate_((X_test_A, X_test_B), y_test_pair, verbose=0)
print(f"Subclassing API Test MSE After Training: {loss_after:.2f}")

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Training and Validation Loss")
plt.show()
