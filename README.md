# MicroFlare — Minimal Deep Learning Framework

**MicroFlare** is a minimalistic deep learning framework implemented in pure Python with a focus on simplicity and learning. It mimics core PyTorch APIs in a single file for educational and experimental purposes.

> **Warning:**  
> This framework is for learning and experimentation only. Do **NOT** use it for production.

---

## Features

- Tensor and automatic differentiation via `Value` and `Tensor` classes  
- Neural network layers: `Linear`, `Conv1d`, `Conv2d`  
- Activation functions: `ReLU`, `Sigmoid`, `Tanh`  
- Recurrent layers: `RNN`, `LSTM`  
- Transformer layers: `Transformer`, `TransformerEncoder`, `TransformerDecoder`
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

- LSTM: Long Short-Term Memory layer with input, forget, cell, and output gates.

- Transformer: Complete transformer architecture with encoder and decoder.

- TransformerEncoder: Multi-layer encoder with self-attention and feed-forward networks.

- TransformerDecoder: Multi-layer decoder with self-attention, cross-attention, and feed-forward networks.

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
### LSTM Layer
```python
import microflare

# Create LSTM layer
lstm = microflare.LSTM(input_size=10, hidden_size=20)

# Input shape: (sequence_length, input_size)
x = microflare.randn((5, 10))

# Forward pass through LSTM
output = lstm(x)  # output shape: (sequence_length, hidden_size)

# Get LSTM parameters for optimization
params = lstm.parameters()
optimizer = microflare.Adam(params, lr=0.001)
```

### Transformer Architecture
```python
import microflare

# Create transformer with encoder-decoder architecture
transformer = microflare.Transformer(
    d_model=512,              # embedding dimension
    num_heads=8,              # number of attention heads
    d_ff=2048,                # feed-forward hidden dimension
    num_encoder_layers=6,     # number of encoder stacks
    num_decoder_layers=6      # number of decoder stacks
)

# Source and target sequences
# Shape: (batch_size, sequence_length, d_model)
src = microflare.randn((2, 10, 512))
tgt = microflare.randn((2, 8, 512))

# Forward pass
output = transformer(src, tgt)  # output shape: (2, 8, 512)

# Get all transformer parameters
params = transformer.parameters()
optimizer = microflare.Adam(params, lr=0.0001)
```

### TransformerEncoder (Standalone)
```python
import microflare

# Create encoder
encoder = microflare.TransformerEncoder(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

# Input shape: (batch_size, sequence_length, d_model)
x = microflare.randn((2, 10, 512))

# Encode sequence
encoded = encoder(x)
print(encoded.shape)  # (2, 10, 512)
```

### TransformerDecoder (Standalone with Encoder Output)
```python
import microflare

# Create encoder and decoder
encoder = microflare.TransformerEncoder(512, 8, 2048, 6)
decoder = microflare.TransformerDecoder(512, 8, 2048, 6)

# Encode source
src = microflare.randn((2, 10, 512))
encoder_output = encoder(src)

# Decode with encoder output
tgt = microflare.randn((2, 8, 512))
decoded = decoder(tgt, encoder_output)
print(decoded.shape)  # (2, 8, 512)
```
````
