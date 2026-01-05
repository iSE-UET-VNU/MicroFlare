# MicroFlare - Minimal Deep Learning Framework

**MicroFlare** is a minimalistic deep learning framework implemented in pure Python with a focus on simplicity and learning. It mimics core PyTorch APIs in a single file for educational and experimental purposes.

> **Warning:**  
> This framework is for learning and experimentation only. Do **NOT** use it for production.

---

## Features

- Tensor and automatic differentiation via `Value` and `Tensor` classes  
- Neural network layers: `Linear`, `Conv1d`, `Conv2d`, `Embedding`
- Activation functions: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Swish`, `LeakyReLU`  
- Normalization layers: `BatchNorm1d`, `LayerNorm`  
- Pooling layers: `MaxPool1d`, `AvgPool1d`
- Recurrent layers: `RNN`, `LSTM`  
- Transformer layers: `Transformer`, `TransformerEncoder`, `TransformerDecoder`
- Loss functions: `MSELoss`, `L1Loss`, `L2Loss`, `HuberLoss`, `CrossEntropyLoss`, `BCELoss`, `KLDivLoss`, `SmoothL1Loss`
- Optimizers: `SGD`, `Adam`, `AdamW`, `RMSprop`
- Utility modules: `Sequential`, `DropOut`, `Flatten`, `Identity`, `Module`
- Tensor operations: reshape, view, transpose, flatten, squeeze, unsqueeze, permute, repeat, mean, std, clamp
- Utility functions: `zeros`, `ones`, `full`, `eye`, `randn`, `arange`, `cat`, `stack`, `one_hot`
- Data utilities: `DataLoader`, gradient clipping functions  

---

## Installation

Clone or download the repository and import the `microflare` module into your project or you can just download it straight from pypi:

```bash
pip install microflare
```

---

## GPU Acceleration with CuPy

MicroFlare supports GPU acceleration via **CuPy**. When available, it automatically uses GPU for computations. The framework gracefully falls back to CPU (NumPy) if CuPy is not installed.

### Installation with GPU Support

```bash
# Install MicroFlare
pip install microflare

# Install CuPy (requires CUDA Toolkit)
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x

# Check your CUDA version:
nvcc --version
```

### Device Management

```python
import microflare

# Check available device
print(microflare.get_device())  # Returns: cpu or gpu

# Automatically detect and use GPU if available
# (default behavior - happens on import)

# Switch device at runtime
microflare.set_device("gpu")  # Force GPU (if CuPy available)
microflare.set_device("cpu")  # Force CPU
microflare.set_device("auto") # Auto-detect (default)

# Move tensors to specific device
x = microflare.randn((100, 100))
x.to("gpu")  # Move to GPU
x.cpu()      # Move back to CPU
x.gpu()      # Move to GPU again

# Check current device
print(x.device())  # Returns: "cpu" or "gpu"
```

### GPU Training Example

```python
import microflare

# Enable GPU
microflare.set_device("gpu")

# Create model (weights automatically on GPU)
model = microflare.Sequential(
    microflare.Linear(784, 128),
    microflare.ReLU(),
    microflare.Linear(128, 10),
)

# Training data on GPU
X_train = microflare.randn((1000, 784)).to("gpu")
y_train = microflare.randint((1000,), 0, 10).to("gpu")

# Create optimizer
optimizer = microflare.Adam(model.parameters(), lr=0.001)
loss_fn = microflare.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    # Forward pass (runs on GPU)
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    
    # Backward pass (GPU computation)
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

### Performance Notes

- **GPU Benefits**: Significant speedup for large models and datasets (>10,000 parameters)
- **CPU Better For**: Small models, prototyping, debugging
- **Memory**: GPU memory is limited; reduce batch size if you hit CUDA out-of-memory errors
- **Auto-fallback**: Framework works without CuPy; no code changes needed

### Troubleshooting GPU Issues

```python
# Check if GPU is available
import microflare
if "gpu" in str(microflare.get_device()):
    print("GPU is available!")
else:
    print("Running on CPU")

# If GPU not detected but you have CUDA:
# 1. Verify CUDA installation: nvcc --version
# 2. Reinstall CuPy with correct CUDA version
# 3. Check CUDA_HOME environment variable

# Force CPU if GPU gives errors
microflare.set_device("cpu")
```

----
## Components Overview

- **Tensor**: Core multi-dimensional array supporting automatic differentiation, supports operations like view, flatten, transpose, squeeze, unsqueeze, mean, std, clamp, etc.

- **Value**: Scalar wrapper supporting computation graph and backpropagation with derivatives for 20+ operations.

- **Module**: Base class for all neural network modules (similar to `nn.Module` in PyTorch) with train/eval modes.

- **Linear**: Fully connected layer with optional bias.

- **Conv1d, Conv2d**: 1D and 2D convolution layers with configurable stride, padding, and kernels.

- **Embedding**: Embedding layer for converting indices to dense vectors.

- **Activation Functions**: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Swish`, `LeakyReLU` as callable modules.

- **Normalization**: `BatchNorm1d` with running statistics, `LayerNorm` for stable training.

- **Pooling**: `MaxPool1d`, `AvgPool1d` with configurable kernel size, stride, and padding.

- **RNN**: Basic recurrent neural network layer with hidden state tracking.

- **LSTM**: Long Short-Term Memory with input, forget, cell, and output gates.

- **Transformer**: Complete transformer architecture combining encoder and decoder.

- **TransformerEncoder**: Multi-layer encoder with scaled dot-product attention and feed-forward networks.

- **TransformerDecoder**: Multi-layer decoder with self-attention, cross-attention, and feed-forward networks.

- **Sequential**: Container to chain layers and modules sequentially.

- **DropOut**: Dropout regularization layer for training.

- **Flatten**: Flatten layer for reshaping multi-dimensional inputs to 2D.

- **Identity**: Pass-through layer that returns input unchanged.

- **Loss Functions**: `MSELoss`, `L1Loss`, `L2Loss`, `HuberLoss`, `CrossEntropyLoss`, `BCELoss`, `KLDivLoss`, `SmoothL1Loss`.

- **Optimizers**: `SGD`, `Adam`, `AdamW` (with weight decay), `RMSprop`.

- **Utilities**: Tensor creation (`zeros`, `ones`, `full`, `eye`, `randn`, `arange`), tensor manipulation (`cat`, `stack`, `one_hot`), gradient clipping, `DataLoader` for batching.
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

### Batch Normalization and Layer Normalization
```python
import microflare

# Batch Normalization
bn = microflare.BatchNorm1d(num_features=64)
x = microflare.randn((32, 64))
x_normalized = bn(x)

# Layer Normalization
ln = microflare.LayerNorm(normalized_shape=64)
x = microflare.randn((32, 64))
x_normalized = ln(x)
```

### Pooling Layers
```python
import microflare

# Max Pooling
max_pool = microflare.MaxPool1d(kernel_size=3, stride=2, padding=1)
x = microflare.randn((2, 3, 10))  # (batch, channels, length)
pooled = max_pool(x)

# Average Pooling
avg_pool = microflare.AvgPool1d(kernel_size=3, stride=2)
x = microflare.randn((2, 3, 10))
pooled = avg_pool(x)
```

### Embedding Layer
```python
import microflare

# Create embedding layer
embed = microflare.Embedding(num_embeddings=1000, embedding_dim=128)

# Convert indices to embeddings
indices = [0, 5, 10, 15]
embeddings = embed(indices)
print(embeddings.shape)  # (4, 128)
```

### Advanced Loss Functions
```python
import microflare

# Cross Entropy Loss (for classification)
predictions = microflare.randn((32, 10))
targets = microflare.Tensor([int(i % 10) for i in range(32)])
loss = microflare.CrossEntropyLoss()(predictions, targets)

# Binary Cross Entropy Loss
probs = microflare.randn((32, 1))
targets = microflare.Tensor([i % 2 for i in range(32)])
loss = microflare.BCELoss()(probs, targets)

# KL Divergence Loss
log_probs = microflare.log_softmax(predictions, dim=1)
target_dist = microflare.randn((32, 10))
loss = microflare.KLDivLoss()(log_probs, target_dist)
```

### Optimizers with Features
```python
import microflare

model = microflare.Sequential(
    microflare.Linear(10, 32),
    microflare.ReLU(),
    microflare.Linear(32, 1),
)

# Using AdamW optimizer with weight decay
optimizer = microflare.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Using RMSprop optimizer
optimizer = microflare.RMSprop(model.parameters(), lr=0.001)

# Gradient clipping
for epoch in range(10):
    x = microflare.randn((32, 10))
    y = microflare.randn((32, 1))
    
    pred = model(x)
    loss = microflare.MSELoss(pred, y)
    
    loss.backward()
    
    # Clip gradients by norm
    microflare.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
```

### Tensor Operations
```python
import microflare

x = microflare.randn((2, 3, 4))

# Flatten
x_flat = x.flatten(start_dim=1)  # shape: (2, 12)

# Squeeze and unsqueeze
x_squeezed = x_flat.squeeze(dim=0)
x_unsqueezed = x_flat.unsqueeze(dim=0)

# Permute dimensions
x_permuted = x.permute(2, 0, 1)  # shape: (4, 2, 3)

# Repeat elements
x_repeated = x.repeat(2, 1, 1)  # shape: (4, 3, 4)

# Concatenate tensors
y = microflare.randn((2, 3, 4))
z = microflare.cat([x, y], dim=0)  # shape: (4, 3, 4)

# Stack tensors
stacked = microflare.stack([x, y], dim=0)  # shape: (2, 2, 3, 4)

# Clamp values
x_clamped = x.clamp(min_val=-1.0, max_val=1.0)

# Statistics
mean = x.mean(dim=0)
std = x.std(dim=0)
```

### Module Base Class and Custom Training Loop
```python
import microflare

model = microflare.Sequential(
    microflare.Linear(10, 32),
    microflare.BatchNorm1d(32),
    microflare.ReLU(),
    microflare.DropOut(p=0.5),
    microflare.Linear(32, 1),
)

# Set to training mode (enables dropout)
model.train()

# Training loop
for epoch in range(100):
    x = microflare.randn((32, 10))
    y = microflare.randn((32, 1))
    
    pred = model(x)
    loss = microflare.MSELoss(pred, y)
    
    loss.backward()
    
    # Manual parameter update
    for p in model.parameters():
        if p.grad is not None:
            p.data -= 0.01 * p.grad
    
    model.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.data:.4f}")

# Set to eval mode (disables dropout, uses running stats for batch norm)
model.eval()
test_pred = model(microflare.randn((32, 10)))
```

### One-Hot Encoding and DataLoader

### Training a Tiny Character-Level Language Model

A complete example of training a simple language model:

```python
import microflare

# Build vocabulary
text = "The quick brown fox jumps. Hello world."
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
data = [char_to_idx[ch] for ch in text]

# LSTM-based Language Model
class TinyLM(microflare.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_size=16):
        super().__init__()
        self.embedding = microflare.Embedding(vocab_size, embed_dim)
        self.lstm = microflare.LSTM(embed_dim, hidden_size)
        self.fc = microflare.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out = self.lstm(embeds)
        last_h = lstm_out.data[-1] if isinstance(lstm_out.data, list) else lstm_out.data
        return self.fc(microflare.Tensor(last_h))
    
    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.lstm.parameters())
        params.extend(self.fc.parameters())
        return params

# Train the model
model = TinyLM(vocab_size)
model.train()
optimizer = microflare.Adam(model.parameters(), lr=0.01)

num_epochs = 30
seq_len = 10

for epoch in range(num_epochs):
    total_loss = 0
    batches = 0
    
    for i in range(0, len(data) - seq_len - 1, seq_len):
        context = data[i:i + seq_len]
        target = data[i + seq_len]
        
        logits = model(microflare.Tensor([context]))
        loss = microflare.CrossEntropyLoss()(logits, microflare.Tensor([target]))
        
        loss.backward()
        microflare.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.data
        batches += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss/batches:.4f}")

# Text generation
def generate(model, start_idx, length=30):
    model.eval()
    generated = [start_idx]
    
    for _ in range(length):
        context = generated[-10:] if len(generated) >= 10 else generated
        while len(context) < 10:
            context = [0] + context
        
        logits = model(microflare.Tensor([context]))
        logits_data = logits.data if not isinstance(logits.data, list) else logits.data[0]
        
        max_idx = 0
        max_val = logits_data[0].data if isinstance(logits_data[0], microflare.Value) else logits_data[0]
        
        for i in range(1, len(logits_data)):
            val = logits_data[i].data if isinstance(logits_data[i], microflare.Value) else logits_data[i]
            if val > max_val:
                max_val = val
                max_idx = i
        
        generated.append(max_idx)
    
    return ''.join([idx_to_char[i] for i in generated if i in idx_to_char])

print("Generated:", generate(model, char_to_idx['T']))
```