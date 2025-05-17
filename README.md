# MicroFlare — Minimal Deep Learning Framework

**MicroFlare** is a minimalistic deep learning framework implemented in pure Python with a focus on simplicity and learning. It mimics core PyTorch APIs in a single file for educational and experimental purposes.

> **Warning:**  
> This framework is for learning and experimentation only. Do **NOT** use it for production.

---

## Features

- Tensor and automatic differentiation via `Value` and `Tensor` classes  
- Neural network layers: `Linear`, `Conv1d`, `Conv2d`  
- Activation functions: `ReLU`, `Sigmoid`, `Tanh`  
- Recurrent layers: `RNN` (LSTM coming soon)  
- Optimizers: `SGD`, `Adam`  
- Loss functions: `MSELoss`, `L1Loss`, `L2Loss`  
- Utility modules: `Sequential`, `DropOut`  

---

## Installation

Clone or download the repository and import the `microflare` module into your project or you can just download it straight from pypi:

```bash
pip install microflare
```
----
## Components Overview

- Tensor: Core multi-dimensional array supporting automatic differentiation.

- Value: Scalar wrapper supporting computation graph and backpropagation.

- Linear: Fully connected layer.

- Conv1d, Conv2d: 1D and 2D convolution layers with stride and padding.

- ReLU, Sigmoid, Tanh: Activation functions as callable modules.

- RNN: Basic recurrent neural network layer.

- Sequential: Container to chain layers and modules sequentially.

- DropOut: Dropout regularization layer.

- Optimizers: SGD and Adam implementations to update parameters.

- Loss functions: Mean Squared Error, L1, and L2 loss functions.
----
## Usage Example:
### Creating a Simple Feedforward Network
```python
import microflare

# Create model with layers and activations
model = microflare.Sequential(
    microflare.Linear(3, 4),
    microflare.ReLU(),
    microflare.Linear(4, 1),
)

# Input tensor of shape (batch_size=2, features=3)
x = microflare.randn((2, 3))

# Forward pass
output = model(x)
print(output)
```
### Training Loop with MSE Loss and SGD Optimizer
```python
import microflare

model = microflare.Sequential(
    microflare.Linear(3, 4),
    microflare.ReLU(),
    microflare.Linear(4, 1),
)

optimizer = microflare.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    inputs = microflare.randn((2, 3))
    targets = microflare.randn((2, 1))

    preds = model(inputs)
    loss = microflare.MSELoss(preds, targets)

    print(f"Epoch {epoch}: Loss = {loss.data:.4f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
### Using Convolutional Layers
```python
conv = microflare.Conv1d(in_channels=1, out_channels=2, kernel_size=3)

# Input shape: (batch_size=2, channels=1, length=10)
x = microflare.randn((2, 1, 10))

output = conv(x)
print(output.shape)
```
----- 
## Tensor Operations
### Creating Tensors
```python
import microflare

# Create a tensor filled with random Gaussian values
x = microflare.randn((3, 4))

# Create a tensor filled with ones
ones = microflare.ones((2, 3))
```
### Basic Arithmetic Operations
```python
a = microflare.randn((2, 3))
b = microflare.randn((2, 3))

c = a + b       # element-wise addition
d = a - b       # element-wise subtraction
e = a * b       # element-wise multiplication
f = a / b       # element-wise division
g = a ** 2      # element-wise power
```
### Matrix Multiplication and Dot Product
```python
a = microflare.randn((2, 3))
b = microflare.randn((3, 4))

c = a @ b       # matrix multiplication (2x4 tensor)

v1 = microflare.randn((3,))
v2 = microflare.randn((3,))

dot_product = v1.dot(v2)  # scalar Value
print(dot_product)
```
### Transpose and Reshape
```python
a = microflare.randn((2, 3))
a_t = a.transpose()  # transpose to (3, 2)

b = microflare.randn((2, 3, 4))
b_view = b.view(6, 4)  # reshape to (6, 4)
```
### Activation Functions as Modules
```python
relu = microflare.ReLU()
sigmoid = microflare.Sigmoid()
tanh = microflare.Tanh()

x = microflare.randn((2, 3))

print(relu(x))
print(sigmoid(x))
print(tanh(x))
```
### DropOut Usage
```python
dropout = microflare.DropOut(p=0.5)
x = microflare.randn((2, 3))
x_dropped = dropout(x)
print(x_dropped)
```