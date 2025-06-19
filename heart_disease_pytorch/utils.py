# utils.py

import torch
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, dl):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds = torch.sigmoid(model(xb))
            all_preds.append(preds.round())
            all_labels.append(yb)
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
