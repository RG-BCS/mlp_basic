# train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from model import build_functional_model

# Load dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=0)

# Normalize features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Create split inputs
X_train_A, X_train_B = X_train_scaled[:, :5], X_train_scaled[:, 2:]
X_valid_A, X_valid_B = X_valid_scaled[:, :5], X_valid_scaled[:, 2:]
X_test_A, X_test_B = X_test_scaled[:, :5], X_test_scaled[:, 2:]

# Build and compile model
model = build_functional_model(input_shape_a=X_train_A.shape[1:], input_shape_b=X_train_B.shape[1:])
model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=1e-3))

# Evaluate before training
loss_before = model.evaluate([X_test_A, X_test_B], y_test, verbose=0)
print(f"Functional API Test MSE Before Training: {loss_before:.2f}")

# Train
history = model.fit(
    [X_train_A, X_train_B], y_train,
    epochs=20,
    batch_size=32,
    validation_data=([X_valid_A, X_valid_B], y_valid),
    verbose=0
)

# Evaluate after training
loss_after = model.evaluate([X_test_A, X_test_B], y_test, verbose=0)
print(f"Functional API Test MSE After Training: {loss_after:.2f}")

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Training and Validation Loss (MSE)")
plt.show()
