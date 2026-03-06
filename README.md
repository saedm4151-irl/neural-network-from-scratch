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
| `training_loop_w_only.py` | Basic training loop with weights only |
| `training_loop_w_and_bias.py` | Training loop with weights and bias |
| `vectorized_training_loop.py` | Optimized vectorized version using NumPy |
| `forward_pass_relu.py` | Forward pass implementation with ReLU |
| `backpropagation.py` | Manual backpropagation from scratch |
| `backpropagation_dataset.py` | Backpropagation applied on a dataset |

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
- [ ] Deeper network with multiple hidden layers

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
