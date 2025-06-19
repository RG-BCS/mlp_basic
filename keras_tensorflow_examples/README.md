# Keras & TensorFlow Deep Learning Examples
```bash
This project demonstrates my ability to build and train deep learning models using **Keras** with a **TensorFlow
backend**. It includes examples using all three major Keras modeling APIs: **Sequential**, **Functional**, and
**Model Subclassing**.

These examples complement the work in my [`mlp_basic`](https://github.com/RG-BCS/mlp_basic/tree/main/mlp_from_scratch)
repository, where I implement neural networks from scratch and compare them with PyTorch implementations.
```
---

## Project Structure

Each folder contains a standalone example with model code, training pipeline, and results.

### `sequential_mnist_fashion/`
```bash
- **Task:** Image classification on Fashion MNIST
- **Model:** Sequential API
- **Highlights:** Simple, linear model stack; great for quick prototyping
```
### `functional_api_california_housing/`
```bash
- **Task:** Regression on the California Housing dataset
- **Model:** Functional API
- **Highlights:** Multiple inputs, flexible architecture design
```
### `subclassing_api_california_housing/`
```bash
- **Task:** Regression on the California Housing dataset
- **Model:** Model Subclassing API
- **Highlights:** Full control of forward pass logic, custom training/evaluation
```
---

## How to Run

```bash
Each example can be run independently:

cd sequential_mnist_fashion
python train.py

cd functional_api_california_housing
python train.py

cd subclassing_api_california_housing
python train.py
```
---
## Requirements

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```
---

## Goals of this Repo

```bash
. Showcase versatility across Keras APIs
. Demonstrate working knowledge of TensorFlow's model-building backends
. Complement PyTorch and from-scratch implementations in separate repos
. Provide clean, reproducible code examples for training neural networks
```
