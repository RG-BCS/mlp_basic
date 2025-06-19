# train.py

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import HeartDiseaseMLP
import matplotlib.pyplot as plt

# 1. Load and prepare data
df = pd.read_csv("heart_cleveland_upload.csv")
y = df.pop('condition').values
X = df.values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)

train_dl = DataLoader(train_ds, batch_size=20, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=10)

# 2. Build model
input_dim = X.shape[1]
model = HeartDiseaseMLP(input_dim)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Training loop
num_epochs = 100
history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

for epoch in range(num_epochs):
    # Train
    model.train()
    t_loss, t_correct = 0.0, 0
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        t_loss += loss.item() * len(yb)
        t_correct += ((pred.sigmoid() > 0.5) == yb).sum().item()

    t_loss /= len(train_dl.dataset)
    t_acc = t_correct / len(train_dl.dataset)

    # Validate
    model.eval()
    v_loss, v_correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            v_loss += loss.item() * len(yb)
            v_correct += ((pred.sigmoid() > 0.5) == yb).sum().item()

    v_loss /= len(valid_dl.dataset)
    v_acc = v_correct / len(valid_dl.dataset)

    history['train_loss'].append(t_loss)
    history['train_acc'].append(t_acc)
    history['valid_loss'].append(v_loss)
    history['valid_acc'].append(v_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} — "
              f"Train Acc: {t_acc:.2%}, Loss: {t_loss:.4f} | "
              f"Valid Acc: {v_acc:.2%}, Loss: {v_loss:.4f}")

# 4. Plot training history
df_hist = pd.DataFrame(history)
df_hist.plot(figsize=(8, 5))
plt.grid()
plt.suptitle("Training & Validation Accuracy / Loss")
plt.show()
