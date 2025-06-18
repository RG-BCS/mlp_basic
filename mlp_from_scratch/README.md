# MLP from Scratch: Classifying MNIST Digits

```bash
This project implements a **Multi-Layer Perceptron (MLP)** from scratch using **NumPy**,
and compares it to an equivalent **PyTorch implementation** — both trained on the MNIST
handwritten digit dataset.
```

---

## Project Highlights

```bash
- Neural network (MLP) implemented **from scratch**
- PyTorch equivalent for comparison
- Visualizations of accuracy and loss
- Final test accuracy:
    - **Custom NumPy model:** ~94.88%
    - **PyTorch model:** ~90.63%

```
---

## Tech Stack

```bash
    | Component      | Library       |
    |----------------|----------------|
    | Dataset        | `scikit-learn` (MNIST via `fetch_openml`) |
    | Core Logic     | `NumPy`, `PyTorch` |
    | Visualization  | `matplotlib` |
    | Splitting & Metrics | `scikit-learn` |
    
```
---

##  Repository Structure

```bash
    mlp_from_scracth/
        ├── demo.ipynb              # Notebook walkthrough with plots and conclusion
        ├── demo_script.py          # Script version of the notebook
        ├── requirements.txt        # All dependencies
        ├── utils.py                # Accuracy, loss, plotting, one-hot encoding, collate
        ├── training.py             # Training loops for custom and PyTorch models
        ├── models                  # custom and pytorch models

```
---

## Getting Started

### 1. Install dependencies
```bash
    pip install -r requirements.txt
```
### 2. Train the agent
```bash
    . Use demo_script.py to test a trained agent quickly.
    . Explore demo.ipynb for an interactive walkthrough and analysis.

```
---

## Training Results

```bash

        | Model       | Test Accuracy |
        | ----------- | ------------- |
        | NumPy MLP   | **94.88%**    |
        | PyTorch MLP | 90.63%        |


```
---
![Training Rewards](rewards_vs_episodes.png)
---

## What You’ll Learn

```bash
. How forward and backward propagation are implemented manually
. How to train a neural net with only NumPy
. Differences in training behavior between custom and PyTorch models
. How to structure clean, modular ML projects
```
---

