import numpy as np
import matplotlib.pyplot as plt
import torch

def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        y (array-like): Array of integer labels.
        num_labels (int): Number of classes.
        
    Returns:
        np.ndarray: One-hot encoded matrix.
    """
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

def minibatch_generator(X, y, minibatch_size):
    """
    Generate mini-batches for training.
    
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Labels.
        minibatch_size (int): Size of each mini-batch.
        
    Yields:
        Tuple[np.ndarray, np.ndarray]: Mini-batch of features and labels.
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

def mse_loss(targets, probas, num_labels=10):
    """
    Calculate mean squared error loss between targets and predictions.
    
    Args:
        targets (np.ndarray): True labels.
        probas (np.ndarray): Predicted probabilities.
        num_labels (int): Number of classes.
        
    Returns:
        float: Mean squared error.
    """
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    """
    Calculate accuracy between true labels and predicted labels.
    
    Args:
        targets (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.
        
    Returns:
        float: Accuracy score.
    """
    return np.mean(predicted_labels == targets)

def compute_mse_and_acc(nnet, X, y, custom=True, num_labels=10, minibatch_size=100):
    """
    Compute average MSE loss and accuracy on dataset.
    
    Args:
        nnet: Model object with forward method.
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        custom (bool): Flag for custom model or PyTorch model.
        num_labels (int): Number of classes.
        minibatch_size (int): Mini-batch size.
        
    Returns:
        Tuple[float, float]: (MSE loss, accuracy)
    """
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for j, (features, targets) in enumerate(minibatch_gen):
        if custom:
            _, probas = nnet.forward(features)
        else:
            features = torch.tensor(features, dtype=torch.float32)
            probas = nnet(features).detach().numpy()
            
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
        
    mse = mse / (j + 1)
    acc = correct_pred / num_examples
    return mse, acc

def plot_training_curves(custom_loss, custom_train_acc, custom_valid_acc, 
                         torch_loss, torch_train_acc, torch_valid_acc):
    """
    Plot training loss and accuracy curves for both custom and PyTorch models.
    
    Args:
        custom_loss (list): Custom model training MSE per epoch.
        custom_train_acc (list): Custom model training accuracy per epoch.
        custom_valid_acc (list): Custom model validation accuracy per epoch.
        torch_loss (list): PyTorch model training MSE per epoch.
        torch_train_acc (list): PyTorch model training accuracy per epoch.
        torch_valid_acc (list): PyTorch model validation accuracy per epoch.
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(len(custom_loss)), custom_loss, label='Custom Model Loss')
    plt.title("Custom Model Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(len(custom_train_acc)), custom_train_acc, label='Training Accuracy')
    plt.plot(range(len(custom_valid_acc)), custom_valid_acc, label='Validation Accuracy')
    plt.title("Custom Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(len(torch_loss)), torch_loss, label='PyTorch Model Loss')
    plt.title("PyTorch Model Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(range(len(torch_train_acc)), torch_train_acc, label='Training Accuracy')
    plt.plot(range(len(torch_valid_acc)), torch_valid_acc, label='Validation Accuracy')
    plt.title("PyTorch Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to stack features and labels correctly.
    
    Args:
        batch (list of tuples): Each tuple is (feature, label)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of features and labels
    """
    x_batch, y_batch = zip(*batch)
    x_batch = torch.stack(x_batch).float()  # Ensure features are float32 tensors
    y_batch = torch.stack(y_batch)           # Labels as tensor (int64)
    return x_batch, y_batch


# ===============================
# Additional utils for 2-layer model (softmax + cross entropy)
# ===============================

def softmax(z):
    """
    Softmax activation function with numerical stability.
    
    Args:
        z (np.ndarray): Input logits of shape (batch_size, num_classes)
        
    Returns:
        np.ndarray: Softmax probabilities.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    Compute cross-entropy loss between true labels and predicted probabilities.
    
    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities (after softmax).
        
    Returns:
        float: Average cross-entropy loss over the batch.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)

def compute_ce_and_acc(nnet, X, y, custom=True, num_labels=10, minibatch_size=100):
    """
    Compute average cross-entropy loss and accuracy on dataset.
    
    Args:
        nnet: Model object with forward method.
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        custom (bool): Flag for custom model or PyTorch model.
        num_labels (int): Number of classes.
        minibatch_size (int): Mini-batch size.
        
    Returns:
        Tuple[float, float]: (cross-entropy loss, accuracy)
    """
    ce_loss_sum, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for j, (features, targets) in enumerate(minibatch_gen):
        if custom:
            _, _, probas = nnet.forward(features)
        else:
            features = torch.tensor(features, dtype=torch.float32)
            probas = nnet(features).detach().numpy()
        
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        
        loss = cross_entropy_loss(onehot_targets, probas)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        ce_loss_sum += loss
    
    ce_loss_avg = ce_loss_sum / (j + 1)
    acc = correct_pred / num_examples
    return ce_loss_avg, acc
