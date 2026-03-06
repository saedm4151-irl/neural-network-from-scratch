# neural-network-from-scratch
Building a neural network from scratch in Python — no frameworks, just math. Covers forward pass, backpropagation, vectorization, and training on real data.

A fully hand-coded neural network built in Python without any ML frameworks.
No PyTorch. No TensorFlow. Just NumPy and math.

This project was built to deeply understand what happens inside a neural network
before relying on high-level libraries.

---

## What's Implemented

- Forward pass (weights only)
- Forward pass with weights and bias
- ReLU activation function
- Backpropagation (manual gradient calculation)
- Backpropagation on a dataset
- #Vectorized training loop (NumPy)
- #Training loop with real data

---

## File Structure

| File | Description |
|------|-------------|
| `1_linear_neuron_weights_only.py` | Single neuron training loop, weights only, no bias |
| `2_linear_neuron_weights_and_bias.py` | Adds bias term to the basic neuron |
| `3_vectorized_training_loop.py` | Vectorized version of the training loop |
| `4_forward_pass_relu.py` | Forward pass with ReLU activation: max(0, x) |
| `5_forward_pass_relu_diagram.png` | Hand-drawn diagram of the forward pass for visual understanding |
| `6_backpropagation_single_sample.py` | Manual backpropagation on a single data point |
| `7_backpropagation_dataset_loop.py` | Backpropagation looped over a small hand-coded dataset |
| `8_vectorized_backpropagation.py` | Fully vectorized backpropagation using NumPy |

---

## Concepts Covered

- What a neuron actually does mathematically
- How gradients flow backward through layers
- Why vectorization matters for speed
- The difference between training with and without bias
- How ReLU introduces non-linearity

---

## In Progress

- [ ] Mini-batch training
- [ ] Stochastic Gradient Descent (SGD)
- [ ] Training on a real dataset (MNIST or similar)
- [ ] Moving into PyTorch — see [pytorch-deep-learning](https://github.com/saedm4151-irl/pytorch-deep-learning.git)

---

## Why I Built This

I'm an AI student at the American University of Ras Al Khaimah. I wanted to
understand neural networks at the lowest level before moving into PyTorch and
deep learning frameworks. Every function here is written by hand.

---

## Stack

- Python 3
- NumPy
- PyCharm (virtual environment)

---

## Author

**Muhammed Saed**  
AI Student — American University of Ras Al Khaimah, UAE  
muhammedsaed.uni@gmail.com
