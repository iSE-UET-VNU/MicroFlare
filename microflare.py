from typing import List, Union
import numpy as np
import random
import math
import copy


# https://docs.pytorch.org/docs/stable/torch.html
def arange(start=0, end=0, step=1):
    values = []
    current = start
    if step == 0:
        raise ValueError("step must not be zero")
    if step > 0:
        while current < end:
            values.append(Value(current))
            current += step
    else:
        while current > end:
            values.append(Value(current))
            current += step

    return Tensor(values)


def ones(size):
    def create_ones(shape):
        if len(shape) == 0:
            return Value(1.0)
        return [create_ones(shape[1:]) for _ in range(shape[0])]

    nested_ones = create_ones(size)
    return Tensor(nested_ones)


def randn(size):
    def create_randn(shape):
        if len(shape) == 0:
            return Value(random.gauss(0, 1))
        return [create_randn(shape[1:]) for _ in range(shape[0])]

    nested_randn = create_randn(size)
    return Tensor(nested_randn)


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g}, op='{self._op}', label='{self.label}')"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Value(-self.data, (self,), "neg")

        def _backward():
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __rpow__(self, base):
        out = Value(base**self.data, (self,), "rpow")

        def _backward():
            if base <= 0:
                raise ValueError("Base must be positive for derivative")
            self.grad += math.log(base) * out.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if other.data == 0:
            raise ZeroDivisionError("Lmao Không thể chia cho 0")
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data**2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Value(other) / self

    def __pow__(self, power):
        if isinstance(power, Value):
            out = Value(self.data**power.data, (self, power), "pow")

            def _backward():
                # d(out)/d(self)   = p * x^(p-1)
                # d(out)/d(power) = ln(x) * x^p
                self.grad += power.data * (self.data ** (power.data - 1)) * out.grad
                power.grad += math.log(self.data + 1e-20) * out.data * out.grad

            out._backward = _backward
        else:
            out = Value(self.data**power, (self,), f"**{power}")

            def _backward():
                self.grad += power * (self.data ** (power - 1)) * out.grad

            out._backward = _backward
        return out

    def __abs__(self):
        out = self if self.data >= 0 else -self

        def _backward():
            if self.data > 0:
                self.grad += 1.0 * out.grad
            elif self.data < 0:
                self.grad += -1.0 * out.grad
            else:
                self.grad += 0.0

        out._backward = _backward
        return out

    def __hash__(self):
        return id(self)

    def _get_data(self, other):
        return other.data if isinstance(other, Value) else other

    def __lt__(self, other):
        return self.data < self._get_data(other)

    def __le__(self, other):
        return self.data <= self._get_data(other)

    def __gt__(self, other):
        return self.data > self._get_data(other)

    def __ge__(self, other):
        return self.data >= self._get_data(other)

    def __eq__(self, other):
        return self.data == self._get_data(other)

    def __ne__(self, other):
        return self.data != self._get_data(other)

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-20), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sin(self):
        s = math.sin(self.data)
        out = Value(s, (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def cos(self):
        c = math.cos(self.data)
        out = Value(c, (self,), "cos")

        def _backward():
            self.grad += -math.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def tan(self):
        t = math.tan(self.data)
        out = Value(t, (self,), "tan")

        def _backward():
            self.grad += (1.0 + t**2) * out.grad  # sec^2(x)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def gelu(self):
        sqrt_2_over_pi = math.sqrt(2 / math.pi)
        x = self
        x3 = x**3
        inner = sqrt_2_over_pi * (x + 0.044715 * x3)
        tanh_inner = inner.tanh()
        half = Value(0.5)
        one = Value(1.0)
        out = half * x * (one + tanh_inner)
        out.label = "gelu"

        def _backward():
            t = tanh_inner.data
            # lmao typing this makes me question my life choices
            dt_dx = (1 - t**2) * sqrt_2_over_pi * (1 + 3 * 0.044715 * x.data**2)
            dx = 0.5 * (1 + t) + 0.5 * x.data * dt_dx
            x.grad += dx * out.grad

        out._backward = _backward

        return out

    def swish(self):
        s = self.sigmoid()
        out = self * s
        out.label = "swish"

        def _backward():
            self.grad += (s.data + self.data * s.data * (1 - s.data)) * out.grad

        out._backward = _backward

        return out

    def leaky_relu(self, alpha=0.01):
        out = Value(
            self.data if self.data > 0 else alpha * self.data, (self,), "leaky_relu"
        )

        def _backward():
            self.grad += (1.0 if self.data > 0 else alpha) * out.grad

        out._backward = _backward
        return out

    def clone(self):
        return copy.deepcopy(self)

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Tensor:
    def __init__(self, data: Union[List, np.ndarray, Value, float, int]):
        self.data = self._to_value_nested(data)
        self.shape = self._infer_shape(self.data)

    def _to_value_nested(self, x):
        if isinstance(x, Value):
            return x
        elif isinstance(x, (float, int)):
            return Value(x)
        elif isinstance(x, np.ndarray):
            return self._to_value_nested(x.tolist())
        elif isinstance(x, list):
            return [self._to_value_nested(elem) for elem in x]
        else:
            raise TypeError(f"Unsupported type {type(x)}")

    def _infer_shape(self, x):
        if isinstance(x, Value):
            return ()
        elif isinstance(x, list):
            if len(x) == 0:
                return (0,)
            return (len(x),) + self._infer_shape(x[0])
        else:
            raise TypeError(f"Unsupported type {type(x)}")

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(np.add(self.data, other.data))

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(np.multiply(self.data, other.data))

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(np.subtract(self.data, other.data))

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(np.subtract(other.data, self.data))

    def __rpow__(self, base):
        if not isinstance(base, Tensor):
            base = Tensor(base)

        def pow_recursive(b, x):
            if isinstance(x, Value) and isinstance(b, Value):
                return b**x
            elif isinstance(x, list) and isinstance(b, list):
                return [pow_recursive(b_elem, x_elem) for b_elem, x_elem in zip(b, x)]
            elif isinstance(x, list) and isinstance(b, Value):
                return [pow_recursive(b, x_elem) for x_elem in x]
            elif isinstance(x, Value) and isinstance(b, list):
                return [pow_recursive(b_elem, x) for b_elem in b]
            else:
                raise TypeError(f"Unsupported types {type(b)}, {type(x)} in __rpow__")

        result_data = pow_recursive(base.data, self.data)
        return Tensor(result_data)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        def div_recursive(a, b):
            if isinstance(a, Value) and isinstance(b, Value):
                return a / b
            elif isinstance(a, list) and isinstance(b, list):
                return [div_recursive(x, y) for x, y in zip(a, b)]
            elif isinstance(a, list) and isinstance(b, Value):
                return [div_recursive(x, b) for x in a]
            elif isinstance(a, Value) and isinstance(b, list):
                return [div_recursive(a, y) for y in b]
            else:
                raise TypeError(
                    f"Unsupported types {type(a)}, {type(b)} in __truediv__"
                )

        result_data = div_recursive(self.data, other.data)
        return Tensor(result_data)

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        def div_recursive(a, b):
            if isinstance(a, Value) and isinstance(b, Value):
                return a / b
            elif isinstance(a, list) and isinstance(b, list):
                return [div_recursive(x, y) for x, y in zip(a, b)]
            elif isinstance(a, list) and isinstance(b, Value):
                return [div_recursive(x, b) for x in a]
            elif isinstance(a, Value) and isinstance(b, list):
                return [div_recursive(a, y) for y in b]
            else:
                raise TypeError(
                    f"Unsupported types {type(a)}, {type(b)} in __rtruediv__"
                )

        result_data = div_recursive(other.data, self.data)
        return Tensor(result_data)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            # TODO
            # Khi np.power(self.data, other.data) thi segfault ?
            raise NotImplementedError("Tensor power Tensor is not implemented yet")
        return Tensor(np.power(self.data, other))

    def __getitem__(self, index):
        result = self.data[index]

        if isinstance(result, list):
            return Tensor(result)
        else:
            return result

    def clone(self):
        return copy.deepcopy(self)

    def abs(self):
        def _abs_recursive(x):
            if isinstance(x, Value):
                return abs(x)
            elif isinstance(x, list):
                return [_abs_recursive(elem) for elem in x]
            else:
                raise TypeError(f"Unsupported type {type(x)} in abs")

        return Tensor(_abs_recursive(self.data))

    def backward(self):
        def _backward_recursive(x):
            if isinstance(x, Value):
                x.backward()
            elif isinstance(x, list):
                for elem in x:
                    _backward_recursive(elem)

        _backward_recursive(self.data)

    def sum(self, dim=None) -> Union["Tensor", Value]:
        # if dim = None => sum of ALL element in the matrix
        def _sum_recursive(x):
            if isinstance(x, Value):
                return x
            elif isinstance(x, list):
                s = _sum_recursive(x[0])
                for elem in x[1:]:
                    s = s + _sum_recursive(elem)
                return s
            else:
                raise TypeError(f"Unsupported type {type(x)} in sum")

        if dim is None:
            return _sum_recursive(self.data)

        if dim == 0:
            rows = len(self.data)
            cols = len(self.data[0]) if rows > 0 else 0

            summed = []
            for c in range(cols):
                s = self.data[0][c]
                for r in range(1, rows):
                    s = s + self.data[r][c]
                summed.append(s)
            return Tensor(summed)

        elif dim == 1:
            summed = []
            for row in self.data:
                s = row[0]
                for elem in row[1:]:
                    s = s + elem
                summed.append(s)
            return Tensor(summed)

        else:
            raise ValueError(
                f"Sum with dim={dim} not supported for tensors with shape {self.shape}"
            )

    def tolist(self):
        def _to_list(x):
            if isinstance(x, Value):
                return x.data
            elif isinstance(x, list):
                return [_to_list(elem) for elem in x]
            else:
                return x

        return _to_list(self.data)

    def _flatten(self, data):
        if isinstance(data, Value):
            return [data]
        elif isinstance(data, list):
            flat = []
            for elem in data:
                flat.extend(self._flatten(elem))
            return flat
        else:
            raise TypeError(f"Unsupported type {type(data)} in _flatten")

    def _reshape(self, flat_data, shape):
        if len(shape) == 0:
            return flat_data[0], flat_data[1:]
        size = shape[0]
        out = []
        rest_shape = shape[1:]
        for _ in range(size):
            elem, flat_data = self._reshape(flat_data, rest_shape)
            out.append(elem)
        return out, flat_data

    def matmul(self, other: "Tensor") -> "Tensor":
        # can not use matmul in numpy LOL, we lose the gradient tracking
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication only defined for 2D tensors")
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Incompatible shapes {self.shape} and {other.shape} for matmul"
            )

        m, n = self.shape
        n2, p = other.shape
        result_data = []
        for i in range(m):
            row = []
            for j in range(p):
                s = Value(0.0)
                for k in range(n):
                    s += self.data[i][k] * other.data[k][j]
                row.append(s)
            result_data.append(row)
        return Tensor(result_data)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def dot(self, other: "Tensor") -> "Value":
        if len(self.shape) != 1 or len(other.shape) != 1:
            raise ValueError("Dot product is defined only for 1D tensors (vectors)")
        if self.shape[0] != other.shape[0]:
            raise ValueError("Vectors must be same length for dot product")

        result = Value(0.0)
        for i in range(self.shape[0]):
            result += self.data[i] * other.data[i]
        return result

    def view(self, *shape):
        total_elems = 1
        for dim in shape:
            total_elems *= dim
        flat = self._flatten(self.data)
        if total_elems != len(flat):
            raise ValueError(
                f"Cannot view tensor of total size {len(flat)} into shape {shape}"
            )

        reshaped_data, remaining = self._reshape(flat, shape)
        if remaining:
            raise RuntimeError("Reshape leftover elements, should not happen")

        return Tensor(reshaped_data)

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError(
                f"Transpose only supported for 2D tensors, but got shape {self.shape}"
            )
        rows, cols = self.shape
        transposed_data = []
        for c in range(cols):
            row = []
            for r in range(rows):
                row.append(self.data[r][c])
            transposed_data.append(row)
        return Tensor(transposed_data)

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions."""
        if end_dim == -1:
            end_dim = len(self.shape) - 1
        
        # Get the new shape
        new_shape = list(self.shape)
        dims_to_flatten = new_shape[start_dim:end_dim+1]
        flattened_size = math.prod(dims_to_flatten) if dims_to_flatten else 1
        
        new_shape = new_shape[:start_dim] + [flattened_size] + new_shape[end_dim+1:]
        
        return self.view(*new_shape)
    
    def unsqueeze(self, dim=0):
        """Add a dimension of size 1 at the specified position."""
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return self.view(*new_shape)
    
    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        if dim is not None:
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} with size {self.shape[dim]}")
            new_shape = list(self.shape)
            new_shape.pop(dim)
            return self.view(*new_shape)
        else:
            new_shape = [s for s in self.shape if s != 1]
            if len(new_shape) == 0:
                new_shape = [1]
            return self.view(*new_shape)
    
    def repeat(self, *sizes):
        """Repeat tensor along specified dimensions."""
        if len(sizes) != len(self.shape):
            raise ValueError(f"Number of repeats ({len(sizes)}) must match number of dimensions ({len(self.shape)})")
        
        repeated = self.data
        for dim, repeat_count in enumerate(sizes):
            if repeat_count == 1:
                continue
            
            def _repeat_dim(data, d, r):
                if d == 0:
                    return data * r
                else:
                    return [_repeat_dim(elem, d - 1, r) for elem in data]
            
            repeated = _repeat_dim(repeated, dim, repeat_count)
        
        new_shape = tuple(s * r for s, r in zip(self.shape, sizes))
        result = Tensor(repeated)
        result.shape = new_shape
        return result
    
    def permute(self, *dims):
        """Permute tensor dimensions."""
        if len(dims) != len(self.shape):
            raise ValueError(f"Number of dims ({len(dims)}) must match tensor rank ({len(self.shape)})")
        
        # Get new shape
        new_shape = tuple(self.shape[i] for i in dims)
        
        # For simplicity, flatten, then reshape with new shape
        flat = self._flatten(self.data)
        # This is a simplified implementation - a full implementation would reorder elements
        result = Tensor(flat).view(*new_shape)
        return result
    
    def mean(self, dim=None):
        """Compute mean of tensor."""
        if dim is None:
            total = Value(0.0)
            count = 0
            
            def _sum_all(data):
                nonlocal total, count
                if isinstance(data, Value):
                    total += data
                    count += 1
                elif isinstance(data, list):
                    for elem in data:
                        _sum_all(elem)
            
            _sum_all(self.data)
            return total / Value(count)
        else:
            # Mean along specific dimension
            if dim == 0:
                result = []
                rows = len(self.data)
                cols = len(self.data[0]) if rows > 0 else 0
                for c in range(cols):
                    s = self.data[0][c]
                    for r in range(1, rows):
                        s = s + self.data[r][c]
                    result.append(s / Value(rows))
                return Tensor(result)
            elif dim == 1:
                result = []
                for row in self.data:
                    s = row[0]
                    for elem in row[1:]:
                        s = s + elem
                    result.append(s / Value(len(row)))
                return Tensor(result)
            else:
                raise ValueError(f"Mean with dim={dim} not supported")
    
    def std(self, dim=None):
        """Compute standard deviation."""
        if dim is None:
            mean = self.mean(dim)
            
            def _compute_variance(data):
                if isinstance(data, Value):
                    return (data - mean) ** 2
                elif isinstance(data, list):
                    return [_compute_variance(elem) for elem in data]
            
            variance = _compute_variance(self.data)
            var_tensor = Tensor(variance)
            return (var_tensor.mean(dim)) ** 0.5
        else:
            raise NotImplementedError("std with dim not yet implemented")
    
    def clamp(self, min_val=None, max_val=None):
        """Clamp tensor values to a range."""
        def _clamp(value):
            if isinstance(value, Value):
                clamped = value.data
                if min_val is not None:
                    clamped = max(clamped, min_val)
                if max_val is not None:
                    clamped = min(clamped, max_val)
                return Value(clamped)
            elif isinstance(value, list):
                return [_clamp(v) for v in value]
        
        return Tensor(_clamp(self.data))


def L1Loss(predicted: Tensor, target: Tensor) -> Value:
    return (predicted - target).abs().sum() / math.prod(predicted.shape)


def L2Loss(predicted: Tensor, target: Tensor) -> Value:
    diff = predicted - target
    squared = diff * diff
    return squared.sum() / math.prod(predicted.shape)


def MSELoss(predicted: Tensor, target: Tensor) -> Value:
    diff = predicted - target
    sq_diff = diff**2
    return sq_diff.sum() / (np.prod(predicted.shape))


def HuberLoss(predicted: Tensor, target: Tensor, delta=1.0) -> Value:
    """Huber loss (smooth approximation of L1 loss)."""
    diff = predicted - target
    
    def _huber(x):
        if isinstance(x, Value):
            abs_x = abs(x.data)
            if abs_x <= delta:
                return 0.5 * (x.data ** 2)
            else:
                return delta * (abs_x - 0.5 * delta)
        elif isinstance(x, list):
            return sum([_huber(elem) for elem in x])
    
    loss = Value(_huber(diff.data))
    return loss / Value(math.prod(predicted.shape))


def Softmax(values: Tensor, dim=1) -> Tensor:
    tmp = math.e**values
    return tmp / tmp.sum(dim)


def Sigmoid(values: Tensor) -> Tensor:
    neg_values = -values
    exp_neg = math.e**neg_values
    return 1 / (1 + exp_neg)


class Sigmoid(object):
    def __call__(self, x: "Tensor") -> "Tensor":
        def sigmoid_value(v: "Value") -> "Value":
            neg = Value(-v.data)
            exp_neg = math.exp(neg.data)
            s = 1 / (1 + exp_neg)
            return Value(s)

        def sigmoid_recursive(x):
            if isinstance(x, Value):
                return sigmoid_value(x)
            elif isinstance(x, list):
                return [sigmoid_recursive(elem) for elem in x]
            else:
                raise TypeError(f"Unsupported type {type(x)} in Sigmoid")

        sigmoid_data = sigmoid_recursive(x.data)
        return Tensor(sigmoid_data)

    def parameters(self):
        return []


class Tanh(object):
    def __call__(self, x: "Tensor") -> "Tensor":
        def tanh_value(v: "Value") -> "Value":
            t = np.tanh(v.data)
            return Value(t)

        def tanh_recursive(x):
            if isinstance(x, Value):
                return tanh_value(x)
            elif isinstance(x, list):
                return [tanh_recursive(elem) for elem in x]
            else:
                raise TypeError(f"Unsupported type {type(x)} in Tanh")

        tanh_data = tanh_recursive(x.data)
        return Tensor(tanh_data)

    def parameters(self):
        return []


class ReLU(object):
    def __call__(self, x: Tensor) -> Tensor:
        def relu_value(v: "Value") -> "Value":
            return v if v.data > 0 else Value(0.0)

        def relu_recursive(x):
            if isinstance(x, Value):
                return relu_value(x)
            elif isinstance(x, list):
                return [relu_recursive(elem) for elem in x]
            else:
                raise TypeError(f"Unsupported type {type(x)} in ReLU")

        relu_data = relu_recursive(x.data)
        return Tensor(relu_data)

    def parameters(self):
        return []


class Linear(object):
    def __init__(self, in_features, out_features, bias=False):
        self.b = None
        if bias:
            self.b = randn((out_features, 1))
            # out_features = nums of neuron
        self.w = randn((out_features, in_features))

    def parameters(self):
        params = []
        for row in self.w.data:
            params.extend(row)
        if self.b:
            params.extend(self.b.data)
        return params

    def __call__(self, x):
        if self.b:
            return x @ self.w.transpose() + self.b
        return x @ self.w.transpose()


class DropOut(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: "Tensor"):
        mask_data = np.random.rand(*x.shape) > self.p
        mask = Tensor(mask_data)
        dropped_out_data = []
        for i in range(len(x.data)):
            row = []
            for j in range(len(x.data[i])):
                row.append(x.data[i][j] * mask.data[i][j])
            dropped_out_data.append(row)

        return Tensor(dropped_out_data)

    def parameters(self):
        return []


class Sequential(object):
    def __init__(self, *modules):
        self.modules = list(modules)

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params


class Conv1d(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Tensor(
            [
                [
                    [Value(random.uniform(-1, 1)) for _ in range(kernel_size)]
                    for _ in range(in_channels)
                ]
                for _ in range(out_channels)
            ]
        )

        self.bias = Tensor([Value(random.uniform(-1, 1)) for _ in range(out_channels)])

    def __call__(self, x: Tensor) -> Tensor:
        batch_size, in_channels, input_length = x.shape

        padded_length = input_length + 2 * self.padding

        if self.padding > 0:
            zero = Value(0.0)
            padded_data = []
            for b in range(batch_size):
                padded_channels = []
                for c in range(in_channels):
                    padded_seq = (
                        [zero] * self.padding + x.data[b][c] + [zero] * self.padding
                    )
                    padded_channels.append(padded_seq)
                padded_data.append(padded_channels)
            x_padded = Tensor(padded_data)
        else:
            x_padded = x

        output_length = (padded_length - self.kernel_size) // self.stride + 1

        output = []
        for b in range(batch_size):
            out_channels_data = []
            for out_c in range(self.out_channels):
                out_seq = []
                for i in range(0, output_length * self.stride, self.stride):
                    s = Value(0.0)
                    for in_c in range(in_channels):
                        for k in range(self.kernel_size):
                            s += (
                                x_padded.data[b][in_c][i + k]
                                * self.weight.data[out_c][in_c][k]
                            )
                    s += self.bias.data[out_c]
                    out_seq.append(s)
                out_channels_data.append(out_seq)
            output.append(out_channels_data)

        return Tensor(output)

    def parameters(self):
        params = []
        for out_c in range(self.out_channels):
            for in_c in range(self.in_channels):
                params.extend(self.weight.data[out_c][in_c])
            params.append(self.bias.data[out_c])
        return params


class Conv2d(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size

        self.stride = stride
        self.padding = padding

        self.weight = Tensor(
            [
                [
                    [
                        [Value(random.uniform(-1, 1)) for _ in range(self.kernel_width)]
                        for _ in range(self.kernel_height)
                    ]
                    for _ in range(in_channels)
                ]
                for _ in range(out_channels)
            ]
        )

        self.bias = Tensor([Value(random.uniform(-1, 1)) for _ in range(out_channels)])

    def __call__(self, x: "Tensor") -> "Tensor":
        batch_size, in_channels, height, width = x.shape

        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding

        if self.padding > 0:
            zero = Value(0.0)
            padded_data = []
            for b in range(batch_size):
                padded_channels = []
                for c in range(in_channels):
                    channel = x.data[b][c]
                    padded_rows = (
                        [[zero] * padded_width] * self.padding
                        + [
                            [zero] * self.padding + row + [zero] * self.padding
                            for row in channel
                        ]
                        + [[zero] * padded_width] * self.padding
                    )
                    padded_channels.append(padded_rows)
                padded_data.append(padded_channels)
            x_padded = Tensor(padded_data)
        else:
            x_padded = x

        out_height = (padded_height - self.kernel_height) // self.stride + 1
        out_width = (padded_width - self.kernel_width) // self.stride + 1

        output = []
        for b in range(batch_size):
            out_channels_data = []
            for out_c in range(self.out_channels):
                out_map = []
                for i in range(0, out_height * self.stride, self.stride):
                    row = []
                    for j in range(0, out_width * self.stride, self.stride):
                        s = Value(0.0)
                        for in_c in range(in_channels):
                            for ki in range(self.kernel_height):
                                for kj in range(self.kernel_width):
                                    s += (
                                        x_padded.data[b][in_c][i + ki][j + kj]
                                        * self.weight.data[out_c][in_c][ki][kj]
                                    )
                        s += self.bias.data[out_c]
                        row.append(s)
                    out_map.append(row)
                out_channels_data.append(out_map)
            output.append(out_channels_data)

        return Tensor(output)

    def parameters(self):
        params = []
        for out_c in range(self.out_channels):
            for in_c in range(self.in_channels):
                for kh in range(self.kernel_height):
                    params.extend(self.weight.data[out_c][in_c][kh])
            params.append(self.bias.data[out_c])
        return params


class Flatten(object):
    """Flatten layer - converts multi-dimensional input to 2D (batch_size, features)."""
    
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def __call__(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)
    
    def parameters(self):
        return []


class Identity(object):
    """Identity layer - returns input unchanged."""
    
    def __call__(self, x: Tensor) -> Tensor:
        return x
    
    def parameters(self):
        return []


class RNN(object):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = Tensor(
            [
                [Value(random.uniform(-1, 1)) for _ in range(input_size)]
                for _ in range(hidden_size)
            ]
        )
        self.W_hh = Tensor(
            [
                [Value(random.uniform(-1, 1)) for _ in range(hidden_size)]
                for _ in range(hidden_size)
            ]
        )
        self.b_ih = Tensor([Value(0.0) for _ in range(hidden_size)])
        self.b_hh = Tensor([Value(0.0) for _ in range(hidden_size)])

    def __call__(self, x: Tensor, h_prev=None):
        seq_len, input_size = x.shape

        if h_prev is None:
            h_prev = Tensor([Value(0.0) for _ in range(self.hidden_size)])

        hidden_states = []
        h_t = h_prev

        for t in range(seq_len):
            x_t = x[t]

            i2h = []
            for i in range(self.hidden_size):
                s = Value(0.0)
                for j in range(self.input_size):
                    s += self.W_ih.data[i][j] * x_t.data[j]
                s += self.b_ih.data[i]
                i2h.append(s)

            h2h = []
            for i in range(self.hidden_size):
                s = Value(0.0)
                for j in range(self.hidden_size):
                    s += self.W_hh.data[i][j] * h_t.data[j]
                s += self.b_hh.data[i]
                h2h.append(s)

            h_new_data = []
            for i in range(self.hidden_size):
                h_new_data.append((i2h[i] + h2h[i]).tanh())
            h_t = Tensor(h_new_data)

            hidden_states.append(h_t)

        return Tensor([h.data for h in hidden_states])

    def parameters(self):
        params = []
        for row in self.W_ih.data:
            params.extend(row)
        for row in self.W_hh.data:
            params.extend(row)
        params.extend(self.b_ih.data)
        params.extend(self.b_hh.data)
        return params


class LSTM(object):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights for input, forget, cell, and output gates
        self.W_ii = Tensor([[Value(random.uniform(-1, 1)) for _ in range(input_size)] for _ in range(hidden_size)])
        self.W_if = Tensor([[Value(random.uniform(-1, 1)) for _ in range(input_size)] for _ in range(hidden_size)])
        self.W_ig = Tensor([[Value(random.uniform(-1, 1)) for _ in range(input_size)] for _ in range(hidden_size)])
        self.W_io = Tensor([[Value(random.uniform(-1, 1)) for _ in range(input_size)] for _ in range(hidden_size)])
        
        # Hidden-to-hidden weights for input, forget, cell, and output gates
        self.W_hi = Tensor([[Value(random.uniform(-1, 1)) for _ in range(hidden_size)] for _ in range(hidden_size)])
        self.W_hf = Tensor([[Value(random.uniform(-1, 1)) for _ in range(hidden_size)] for _ in range(hidden_size)])
        self.W_hg = Tensor([[Value(random.uniform(-1, 1)) for _ in range(hidden_size)] for _ in range(hidden_size)])
        self.W_ho = Tensor([[Value(random.uniform(-1, 1)) for _ in range(hidden_size)] for _ in range(hidden_size)])
        
        # Biases for all gates
        self.b_i = Tensor([Value(0.0) for _ in range(hidden_size)])
        self.b_f = Tensor([Value(0.0) for _ in range(hidden_size)])
        self.b_g = Tensor([Value(0.0) for _ in range(hidden_size)])
        self.b_o = Tensor([Value(0.0) for _ in range(hidden_size)])

    def __call__(self, x: "Tensor", h_prev=None, c_prev=None):
        seq_len, input_size = x.shape
        
        if h_prev is None:
            h_prev = Tensor([Value(0.0) for _ in range(self.hidden_size)])
        if c_prev is None:
            c_prev = Tensor([Value(0.0) for _ in range(self.hidden_size)])
        
        hidden_states = []
        h_t = h_prev
        c_t = c_prev
        
        for t in range(seq_len):
            x_t = x[t]
            
            # Input gate
            i_t_data = []
            for i in range(self.hidden_size):
                s = Value(0.0)
                for j in range(self.input_size):
                    s += self.W_ii.data[i][j] * x_t.data[j]
                for j in range(self.hidden_size):
                    s += self.W_hi.data[i][j] * h_t.data[j]
                s += self.b_i.data[i]
                i_t_data.append(s.sigmoid())
            i_t = Tensor(i_t_data)
            
            # Forget gate
            f_t_data = []
            for i in range(self.hidden_size):
                s = Value(0.0)
                for j in range(self.input_size):
                    s += self.W_if.data[i][j] * x_t.data[j]
                for j in range(self.hidden_size):
                    s += self.W_hf.data[i][j] * h_t.data[j]
                s += self.b_f.data[i]
                f_t_data.append(s.sigmoid())
            f_t = Tensor(f_t_data)
            
            # Cell gate (candidate values)
            g_t_data = []
            for i in range(self.hidden_size):
                s = Value(0.0)
                for j in range(self.input_size):
                    s += self.W_ig.data[i][j] * x_t.data[j]
                for j in range(self.hidden_size):
                    s += self.W_hg.data[i][j] * h_t.data[j]
                s += self.b_g.data[i]
                g_t_data.append(s.tanh())
            g_t = Tensor(g_t_data)
            
            # Cell state update
            c_t_data = []
            for i in range(self.hidden_size):
                c_t_data.append(f_t.data[i] * c_t.data[i] + i_t.data[i] * g_t.data[i])
            c_t = Tensor(c_t_data)
            
            # Output gate
            o_t_data = []
            for i in range(self.hidden_size):
                s = Value(0.0)
                for j in range(self.input_size):
                    s += self.W_io.data[i][j] * x_t.data[j]
                for j in range(self.hidden_size):
                    s += self.W_ho.data[i][j] * h_t.data[j]
                s += self.b_o.data[i]
                o_t_data.append(s.sigmoid())
            o_t = Tensor(o_t_data)
            
            # Hidden state update
            h_t_data = []
            for i in range(self.hidden_size):
                h_t_data.append(o_t.data[i] * c_t.data[i].tanh())
            h_t = Tensor(h_t_data)
            
            hidden_states.append(h_t)
        
        return Tensor([h.data for h in hidden_states])

    def parameters(self):
        params = []
        for row in self.W_ii.data:
            params.extend(row)
        for row in self.W_if.data:
            params.extend(row)
        for row in self.W_ig.data:
            params.extend(row)
        for row in self.W_io.data:
            params.extend(row)
        for row in self.W_hi.data:
            params.extend(row)
        for row in self.W_hf.data:
            params.extend(row)
        for row in self.W_hg.data:
            params.extend(row)
        for row in self.W_ho.data:
            params.extend(row)
        params.extend(self.b_i.data)
        params.extend(self.b_f.data)
        params.extend(self.b_g.data)
        params.extend(self.b_o.data)
        return params


class Transformer(object):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_encoder_layers=6, num_decoder_layers=6):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Encoder and decoder components
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_decoder_layers)
        
        # Output projection layer
        self.output_projection = Linear(d_model, d_model, bias=False)

    def __call__(self, src: Tensor, tgt: Tensor, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass through transformer.
        
        Args:
            src: Source sequence (batch_size, src_seq_len, d_model)
            tgt: Target sequence (batch_size, tgt_seq_len, d_model)
            src_mask: Mask for source sequence
            tgt_mask: Mask for target sequence (causal mask for autoregressive)
            memory_mask: Mask for cross-attention
        
        Returns:
            output: Decoder output (batch_size, tgt_seq_len, d_model)
        """
        # Encode source
        encoder_output = self.encoder(src, src_mask)
        
        # Decode with encoder output
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)
        
        # Project to output dimension
        output = self.output_projection(decoder_output)
        
        return output

    def parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        params.extend(self.output_projection.parameters())
        return params


class TransformerEncoder(object):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Multi-head attention and feed-forward layers for each encoder block
        self.attention_weights = []
        self.feed_forward_weights = []
        self.layer_norms = []
        
        for _ in range(num_layers):
            # Attention weights: Q, K, V projections
            self.attention_weights.append({
                'W_q': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_k': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_v': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_o': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)])
            })
            
            # Feed-forward weights: two linear layers
            self.feed_forward_weights.append({
                'W1': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_ff)]),
                'b1': Tensor([Value(0.0) for _ in range(d_ff)]),
                'W2': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_ff)] for _ in range(d_model)]),
                'b2': Tensor([Value(0.0) for _ in range(d_model)])
            })
            
            # Layer normalization parameters
            self.layer_norms.append({
                'gamma': Tensor([Value(1.0) for _ in range(d_model)]),
                'beta': Tensor([Value(0.0) for _ in range(d_model)])
            })

    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        # Compute attention scores: (Q @ K^T) / sqrt(d_k)
        batch_size, seq_len, d_k = Q.shape
        
        scores_list = []
        for q_seq in Q.data:
            row_scores = []
            for q in q_seq:
                score_row = []
                for k_seq in K.data:
                    s = Value(0.0)
                    for k in k_seq:
                        dot = s + q.data * k.data if isinstance(q, Value) and isinstance(k, Value) else s
                        s = dot
                    score_row.append(s / math.sqrt(d_k))
                row_scores.append(score_row)
            scores_list.append(row_scores)
        
        # Simplified softmax for attention (approximate)
        attention_weights = []
        for scores in scores_list:
            weighted = []
            for score_row in scores:
                max_score = max([s.data for s in score_row]) if score_row else 0
                exp_scores = [Value(math.exp(s.data - max_score)) for s in score_row]
                sum_exp = sum([e.data for e in exp_scores])
                weights = [e / Value(sum_exp) for e in exp_scores]
                weighted.append(weights)
            attention_weights.append(weighted)
        
        # Apply attention to values
        output = []
        for b, attn_weights in enumerate(attention_weights):
            out_seq = []
            for pos, weight_row in enumerate(attn_weights):
                weighted_v = Value(0.0)
                for t, w in enumerate(weight_row):
                    weighted_v += w * V.data[b][t]
                out_seq.append(weighted_v)
            output.append(out_seq)
        
        return Tensor(output)

    def __call__(self, x: Tensor, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        output = x
        
        for layer_idx in range(self.num_layers):
            # Multi-head attention (simplified to single head)
            attn_weights = self.attention_weights[layer_idx]
            Q = output @ attn_weights['W_q']
            K = output @ attn_weights['W_k']
            V = output @ attn_weights['W_v']
            
            attn_output = self._scaled_dot_product_attention(Q, K, V, mask)
            attn_output = attn_output @ attn_weights['W_o']
            
            # Add & Norm
            output_data = []
            for i, (out_val, attn_val) in enumerate(zip(output.data, attn_output.data)):
                merged = []
                for o, a in zip(out_val, attn_val):
                    merged.append(o + a)
                output_data.append(merged)
            output = Tensor(output_data)
            
            # Feed-forward network
            ff_weights = self.feed_forward_weights[layer_idx]
            ff_hidden = output @ ff_weights['W1']
            
            # Add bias and apply ReLU
            ff_hidden_data = []
            for row in ff_hidden.data:
                new_row = []
                for val in row:
                    new_row.append((val + ff_weights['b1'].data[0]).relu() if isinstance(val, Value) else val)
                ff_hidden_data.append(new_row)
            ff_hidden = Tensor(ff_hidden_data)
            
            ff_output = ff_hidden @ ff_weights['W2']
            
            ff_output_data = []
            for row in ff_output.data:
                new_row = []
                for val, b in zip(row, ff_weights['b2'].data):
                    new_row.append(val + b)
                ff_output_data.append(new_row)
            ff_output = Tensor(ff_output_data)
            
            output_data = []
            for out_val, ff_val in zip(output.data, ff_output.data):
                merged = []
                for o, f in zip(out_val, ff_val):
                    merged.append(o + f)
                output_data.append(merged)
            output = Tensor(output_data)
        
        return output

    def parameters(self):
        params = []
        for layer_attn in self.attention_weights:
            for w_name in ['W_q', 'W_k', 'W_v', 'W_o']:
                for row in layer_attn[w_name].data:
                    params.extend(row)
        
        for layer_ff in self.feed_forward_weights:
            for row in layer_ff['W1'].data:
                params.extend(row)
            params.extend(layer_ff['b1'].data)
            for row in layer_ff['W2'].data:
                params.extend(row)
            params.extend(layer_ff['b2'].data)
        
        for layer_ln in self.layer_norms:
            params.extend(layer_ln['gamma'].data)
            params.extend(layer_ln['beta'].data)
        
        return params


class TransformerDecoder(object):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        self.self_attention_weights = []
        self.cross_attention_weights = []
        self.feed_forward_weights = []
        self.layer_norms = []
        
        for _ in range(num_layers):
            self.self_attention_weights.append({
                'W_q': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_k': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_v': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_o': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)])
            })
            
            self.cross_attention_weights.append({
                'W_q': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_k': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_v': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)]),
                'W_o': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_model)])
            })
            
            self.feed_forward_weights.append({
                'W1': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_model)] for _ in range(d_ff)]),
                'b1': Tensor([Value(0.0) for _ in range(d_ff)]),
                'W2': Tensor([[Value(random.uniform(-1, 1)) for _ in range(d_ff)] for _ in range(d_model)]),
                'b2': Tensor([Value(0.0) for _ in range(d_model)])
            })
            
            self.layer_norms.append({
                'gamma': Tensor([Value(1.0) for _ in range(d_model)]),
                'beta': Tensor([Value(0.0) for _ in range(d_model)])
            })

    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        batch_size, seq_len, d_k = Q.shape
        
        scores_list = []
        for q_seq in Q.data:
            row_scores = []
            for q in q_seq:
                score_row = []
                for k_seq in K.data:
                    s = Value(0.0)
                    for k in k_seq:
                        dot = s + q.data * k.data if isinstance(q, Value) and isinstance(k, Value) else s
                        s = dot
                    score_row.append(s / math.sqrt(d_k))
                row_scores.append(score_row)
            scores_list.append(row_scores)
        
        attention_weights = []
        for scores in scores_list:
            weighted = []
            for score_row in scores:
                max_score = max([s.data for s in score_row]) if score_row else 0
                exp_scores = [Value(math.exp(s.data - max_score)) for s in score_row]
                sum_exp = sum([e.data for e in exp_scores])
                weights = [e / Value(sum_exp) for e in exp_scores]
                weighted.append(weights)
            attention_weights.append(weighted)
        
        output = []
        for b, attn_weights in enumerate(attention_weights):
            out_seq = []
            for pos, weight_row in enumerate(attn_weights):
                weighted_v = Value(0.0)
                for t, w in enumerate(weight_row):
                    weighted_v += w * V.data[b][t]
                out_seq.append(weighted_v)
            output.append(out_seq)
        
        return Tensor(output)

    def __call__(self, x: Tensor, encoder_output: Tensor, self_mask=None, cross_mask=None):
        output = x
        
        for layer_idx in range(self.num_layers):
            # Self-attention
            self_attn_weights = self.self_attention_weights[layer_idx]
            Q_self = output @ self_attn_weights['W_q']
            K_self = output @ self_attn_weights['W_k']
            V_self = output @ self_attn_weights['W_v']
            
            self_attn_output = self._scaled_dot_product_attention(Q_self, K_self, V_self, self_mask)
            self_attn_output = self_attn_output @ self_attn_weights['W_o']
            
            output_data = []
            for out_val, attn_val in zip(output.data, self_attn_output.data):
                merged = []
                for o, a in zip(out_val, attn_val):
                    merged.append(o + a)
                output_data.append(merged)
            output = Tensor(output_data)
            
            cross_attn_weights = self.cross_attention_weights[layer_idx]
            Q_cross = output @ cross_attn_weights['W_q']
            K_cross = encoder_output @ cross_attn_weights['W_k']
            V_cross = encoder_output @ cross_attn_weights['W_v']
            
            cross_attn_output = self._scaled_dot_product_attention(Q_cross, K_cross, V_cross, cross_mask)
            cross_attn_output = cross_attn_output @ cross_attn_weights['W_o']
            
            output_data = []
            for out_val, attn_val in zip(output.data, cross_attn_output.data):
                merged = []
                for o, a in zip(out_val, attn_val):
                    merged.append(o + a)
                output_data.append(merged)
            output = Tensor(output_data)
            
            ff_weights = self.feed_forward_weights[layer_idx]
            ff_hidden = output @ ff_weights['W1']
            
            ff_hidden_data = []
            for row in ff_hidden.data:
                new_row = []
                for val in row:
                    new_row.append((val + ff_weights['b1'].data[0]).relu() if isinstance(val, Value) else val)
                ff_hidden_data.append(new_row)
            ff_hidden = Tensor(ff_hidden_data)
            
            ff_output = ff_hidden @ ff_weights['W2']
            
            ff_output_data = []
            for row in ff_output.data:
                new_row = []
                for val, b in zip(row, ff_weights['b2'].data):
                    new_row.append(val + b)
                ff_output_data.append(new_row)
            ff_output = Tensor(ff_output_data)
            
            output_data = []
            for out_val, ff_val in zip(output.data, ff_output.data):
                merged = []
                for o, f in zip(out_val, ff_val):
                    merged.append(o + f)
                output_data.append(merged)
            output = Tensor(output_data)
        
        return output

    def parameters(self):
        params = []
        for layer_attn in self.self_attention_weights:
            for w_name in ['W_q', 'W_k', 'W_v', 'W_o']:
                for row in layer_attn[w_name].data:
                    params.extend(row)
        
        for layer_attn in self.cross_attention_weights:
            for w_name in ['W_q', 'W_k', 'W_v', 'W_o']:
                for row in layer_attn[w_name].data:
                    params.extend(row)
        
        for layer_ff in self.feed_forward_weights:
            for row in layer_ff['W1'].data:
                params.extend(row)
            params.extend(layer_ff['b1'].data)
            for row in layer_ff['W2'].data:
                params.extend(row)
            params.extend(layer_ff['b2'].data)
        
        for layer_ln in self.layer_norms:
            params.extend(layer_ln['gamma'].data)
            params.extend(layer_ln['beta'].data)
        
        return params


class Module(object):
    """Base class for all neural network modules in MicroFlare (similar to nn.Module in PyTorch)."""
    
    def __init__(self):
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self):
        raise NotImplementedError
    
    def train(self):
        """Set module to training mode."""
        self.training = True
        return self
    
    def eval(self):
        """Set module to evaluation mode."""
        self.training = False
        return self
    
    def zero_grad(self):
        """Zero the gradients of all parameters."""
        for p in self.parameters():
            p.grad = 0.0


class BatchNorm1d(Module):
    """1D Batch Normalization layer."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Tensor([Value(1.0) for _ in range(num_features)])
        self.bias = Tensor([Value(0.0) for _ in range(num_features)])
        
        # Running statistics
        self.running_mean = Tensor([Value(0.0) for _ in range(num_features)])
        self.running_var = Tensor([Value(1.0) for _ in range(num_features)])
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Compute batch statistics
            batch_mean = []
            batch_var = []
            
            for i in range(self.num_features):
                values = []
                for sample in x.data:
                    values.append(sample[i].data)
                mean = sum(values) / len(values)
                variance = sum([(v - mean) ** 2 for v in values]) / len(values)
                batch_mean.append(mean)
                batch_var.append(variance)
            
            # Normalize
            normalized = []
            for sample in x.data:
                norm_sample = []
                for i in range(self.num_features):
                    normalized_val = (sample[i].data - batch_mean[i]) / math.sqrt(batch_var[i] + self.eps)
                    norm_sample.append(Value(normalized_val))
                normalized.append(norm_sample)
            
            # Scale and shift
            output = []
            for sample in normalized:
                scaled_sample = []
                for i in range(self.num_features):
                    scaled_val = self.weight.data[i].data * sample[i].data + self.bias.data[i].data
                    scaled_sample.append(Value(scaled_val))
                output.append(scaled_sample)
            
            # Update running statistics
            for i in range(self.num_features):
                self.running_mean.data[i].data = (1 - self.momentum) * self.running_mean.data[i].data + self.momentum * batch_mean[i]
                self.running_var.data[i].data = (1 - self.momentum) * self.running_var.data[i].data + self.momentum * batch_var[i]
            
            return Tensor(output)
        else:
            # Use running statistics in eval mode
            normalized = []
            for sample in x.data:
                norm_sample = []
                for i in range(self.num_features):
                    normalized_val = (sample[i].data - self.running_mean.data[i].data) / math.sqrt(self.running_var.data[i].data + self.eps)
                    norm_sample.append(Value(normalized_val))
                normalized.append(norm_sample)
            
            # Scale and shift
            output = []
            for sample in normalized:
                scaled_sample = []
                for i in range(self.num_features):
                    scaled_val = self.weight.data[i].data * sample[i].data + self.bias.data[i].data
                    scaled_sample.append(Value(scaled_val))
                output.append(scaled_sample)
            
            return Tensor(output)
    
    def parameters(self):
        params = []
        params.extend(self.weight.data)
        params.extend(self.bias.data)
        return params


class LayerNorm(Module):
    """Layer Normalization."""
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        
        self.weight = Tensor([Value(1.0) for _ in range(normalized_shape)])
        self.bias = Tensor([Value(0.0) for _ in range(normalized_shape)])
    
    def forward(self, x: Tensor) -> Tensor:
        # Compute mean and variance over last dimension
        def _layer_norm_recursive(data, depth=0):
            if isinstance(data, Value):
                return data
            elif isinstance(data, list):
                if depth == len(x.shape) - 1:
                    # Last dimension - compute statistics
                    values = [v.data for v in data]
                    mean = sum(values) / len(values)
                    var = sum([(v - mean) ** 2 for v in values]) / len(values)
                    
                    normalized = []
                    for i, v in enumerate(data):
                        norm_val = (v.data - mean) / math.sqrt(var + self.eps)
                        scaled = norm_val * self.weight.data[i].data + self.bias.data[i].data
                        normalized.append(Value(scaled))
                    return normalized
                else:
                    return [_layer_norm_recursive(elem, depth + 1) for elem in data]
        
        normalized_data = _layer_norm_recursive(x.data)
        return Tensor(normalized_data)
    
    def parameters(self):
        params = []
        params.extend(self.weight.data)
        params.extend(self.bias.data)
        return params


class MaxPool1d(object):
    """1D Max Pooling layer."""
    
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def __call__(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, channels, length)
        batch_size, channels, length = x.shape
        
        padded_length = length + 2 * self.padding
        output_length = (padded_length - self.kernel_size) // self.stride + 1
        
        # Add padding if needed
        if self.padding > 0:
            zero = Value(0.0)
            padded_data = []
            for b in range(batch_size):
                padded_channels = []
                for c in range(channels):
                    padded_seq = [zero] * self.padding + x.data[b][c] + [zero] * self.padding
                    padded_channels.append(padded_seq)
                padded_data.append(padded_channels)
            x_padded = Tensor(padded_data)
        else:
            x_padded = x
        
        # Apply max pooling
        output = []
        for b in range(batch_size):
            out_channels = []
            for c in range(channels):
                out_seq = []
                for i in range(0, output_length * self.stride, self.stride):
                    window = x_padded.data[b][c][i:i + self.kernel_size]
                    max_val = max([v.data for v in window])
                    out_seq.append(Value(max_val))
                out_channels.append(out_seq)
            output.append(out_channels)
        
        return Tensor(output)
    
    def parameters(self):
        return []


class AvgPool1d(object):
    """1D Average Pooling layer."""
    
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def __call__(self, x: Tensor) -> Tensor:
        batch_size, channels, length = x.shape
        
        padded_length = length + 2 * self.padding
        output_length = (padded_length - self.kernel_size) // self.stride + 1
        
        if self.padding > 0:
            zero = Value(0.0)
            padded_data = []
            for b in range(batch_size):
                padded_channels = []
                for c in range(channels):
                    padded_seq = [zero] * self.padding + x.data[b][c] + [zero] * self.padding
                    padded_channels.append(padded_seq)
                padded_data.append(padded_channels)
            x_padded = Tensor(padded_data)
        else:
            x_padded = x
        
        output = []
        for b in range(batch_size):
            out_channels = []
            for c in range(channels):
                out_seq = []
                for i in range(0, output_length * self.stride, self.stride):
                    window = x_padded.data[b][c][i:i + self.kernel_size]
                    avg_val = sum([v.data for v in window]) / len(window)
                    out_seq.append(Value(avg_val))
                out_channels.append(out_seq)
            output.append(out_channels)
        
        return Tensor(output)
    
    def parameters(self):
        return []


class Embedding(object):
    """Embedding layer for converting indices to dense vectors."""
    
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Embedding matrix: (num_embeddings, embedding_dim)
        self.weight = Tensor([
            [Value(random.uniform(-1, 1)) for _ in range(embedding_dim)]
            for _ in range(num_embeddings)
        ])
    
    def __call__(self, indices: Union[Tensor, List[int]]) -> Tensor:
        # Convert indices to embeddings
        if isinstance(indices, Tensor):
            indices_data = self._flatten(indices.data)
            indices_values = [int(v.data) if isinstance(v, Value) else int(v) for v in indices_data]
        else:
            indices_values = indices
        
        embeddings = []
        for idx in indices_values:
            if idx < 0 or idx >= self.num_embeddings:
                raise IndexError(f"Index {idx} out of range for embedding with {self.num_embeddings} embeddings")
            embeddings.append(self.weight.data[idx])
        
        # Reshape to match input shape + embedding_dim
        if isinstance(indices, Tensor):
            original_shape = indices.shape
            output_shape = original_shape + (self.embedding_dim,)
            # Flatten embeddings and reshape
            flat_embeddings = []
            for emb in embeddings:
                flat_embeddings.extend(emb)
            # For simplicity, return as (num_indices, embedding_dim)
            return Tensor([emb for emb in embeddings])
        else:
            return Tensor(embeddings)
    
    def parameters(self):
        params = []
        for row in self.weight.data:
            params.extend(row)
        return params


class CrossEntropyLoss(object):
    """Cross Entropy Loss (combines LogSoftmax and NLLLoss)."""
    
    def __call__(self, logits: Tensor, targets: Tensor) -> Value:
        # logits: (batch_size, num_classes)
        # targets: (batch_size,) with class indices
        
        batch_size = len(logits.data)
        loss = Value(0.0)
        
        for b in range(batch_size):
            # Compute softmax for batch sample
            logits_row = logits.data[b]
            max_logit = max([v.data for v in logits_row])
            
            exp_logits = [math.exp(v.data - max_logit) for v in logits_row]
            sum_exp = sum(exp_logits)
            softmax = [e / sum_exp for e in exp_logits]
            
            # Get target class
            target_idx = int(targets.data[b].data) if isinstance(targets.data[b], Value) else int(targets.data[b])
            
            # Cross entropy: -log(softmax[target])
            ce = -math.log(softmax[target_idx] + 1e-10)
            loss += Value(ce)
        
        return loss / Value(batch_size)


class BCELoss(object):
    """Binary Cross Entropy Loss."""
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Value:
        # predictions: probabilities between 0 and 1
        # targets: binary labels (0 or 1)
        
        def _bce_recursive(pred, tgt):
            if isinstance(pred, Value) and isinstance(tgt, Value):
                p = pred.data
                t = tgt.data
                return -(t * math.log(p + 1e-10) + (1 - t) * math.log(1 - p + 1e-10))
            elif isinstance(pred, list) and isinstance(tgt, list):
                return sum([_bce_recursive(p, t) for p, t in zip(pred, tgt)])
            else:
                raise TypeError(f"Unsupported types in BCELoss")
        
        loss = Value(_bce_recursive(predictions.data, targets.data))
        count = math.prod(predictions.shape)
        return loss / Value(count)


class KLDivLoss(object):
    """Kullback-Leibler Divergence Loss."""
    
    def __call__(self, log_predictions: Tensor, targets: Tensor) -> Value:
        # log_predictions: log probabilities
        # targets: probability distribution
        
        loss = Value(0.0)
        
        def _kld_recursive(log_pred, tgt):
            if isinstance(log_pred, Value) and isinstance(tgt, Value):
                return tgt.data * (math.log(tgt.data + 1e-10) - log_pred.data)
            elif isinstance(log_pred, list) and isinstance(tgt, list):
                return sum([_kld_recursive(lp, t) for lp, t in zip(log_pred, tgt)])
            else:
                raise TypeError(f"Unsupported types in KLDivLoss")
        
        loss = Value(_kld_recursive(log_predictions.data, targets.data))
        count = math.prod(log_predictions.shape)
        return loss / Value(count)


class SmoothL1Loss(object):
    """Smooth L1 Loss (Huber Loss)."""
    
    def __init__(self, beta=1.0):
        self.beta = beta
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Value:
        diff = predictions - targets
        
        def _smooth_l1(x):
            if isinstance(x, Value):
                abs_x = abs(x.data)
                if abs_x < self.beta:
                    return 0.5 * (x.data ** 2) / self.beta
                else:
                    return abs_x - 0.5 * self.beta
            elif isinstance(x, list):
                return sum([_smooth_l1(elem) for elem in x])
            else:
                raise TypeError(f"Unsupported type in SmoothL1Loss")
        
        loss = Value(_smooth_l1(diff.data))
        count = math.prod(predictions.shape)
        return loss / Value(count)


def softmax(x: Tensor, dim: int = 0) -> Tensor:
    """Apply softmax activation along a dimension."""
    if dim == 0:
        # Softmax over first dimension (batch)
        max_val = max([max([v.data for v in row]) for row in x.data])
        exp_data = [[math.exp(v.data - max_val) for v in row] for row in x.data]
        sum_exp = sum([sum(row) for row in exp_data])
        return Tensor([[Value(e / sum_exp) for e in row] for row in exp_data])
    elif dim == 1:
        # Softmax over second dimension (features)
        softmax_data = []
        for row in x.data:
            max_val = max([v.data for v in row])
            exp_row = [math.exp(v.data - max_val) for v in row]
            sum_exp = sum(exp_row)
            softmax_data.append([Value(e / sum_exp) for e in exp_row])
        return Tensor(softmax_data)
    else:
        raise ValueError(f"Softmax with dim={dim} not supported for tensor with shape {x.shape}")


def log_softmax(x: Tensor, dim: int = 0) -> Tensor:
    """Apply log softmax activation along a dimension."""
    softmax_result = softmax(x, dim)
    
    def _log_recursive(data):
        if isinstance(data, Value):
            return Value(math.log(data.data + 1e-10))
        elif isinstance(data, list):
            return [_log_recursive(elem) for elem in data]
    
    return Tensor(_log_recursive(softmax_result.data))


class SGD(object):
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class Adam(object):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = {id(p): 0 for p in params}
        self.v = {id(p): 0 for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad

            self.m[id(p)] = self.beta1 * self.m[id(p)] + (1 - self.beta1) * grad
            self.v[id(p)] = self.beta2 * self.v[id(p)] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[id(p)] / (1 - self.beta1**self.t)
            v_hat = self.v[id(p)] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class RMSprop(object):
    """RMSprop optimizer."""
    
    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = {id(p): 0 for p in params}
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            grad = p.grad
            self.v[id(p)] = self.alpha * self.v[id(p)] + (1 - self.alpha) * (grad ** 2)
            p.data -= self.lr * grad / (self.v[id(p)] ** 0.5 + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class AdamW(object):
    """Adam with Weight Decay (AdamW optimizer)."""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = {id(p): 0 for p in params}
        self.v = {id(p): 0 for p in params}
    
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            
            grad = p.grad
            
            # L2 weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            self.m[id(p)] = self.beta1 * self.m[id(p)] + (1 - self.beta1) * grad
            self.v[id(p)] = self.beta2 * self.v[id(p)] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[id(p)] / (1 - self.beta1 ** self.t)
            v_hat = self.v[id(p)] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


# Utility functions

def zeros(size):
    """Create a tensor filled with zeros."""
    def create_zeros(shape):
        if len(shape) == 0:
            return Value(0.0)
        return [create_zeros(shape[1:]) for _ in range(shape[0])]
    
    nested_zeros = create_zeros(size)
    return Tensor(nested_zeros)


def full(size, fill_value):
    """Create a tensor filled with a specific value."""
    def create_full(shape, value):
        if len(shape) == 0:
            return Value(value)
        return [create_full(shape[1:], value) for _ in range(shape[0])]
    
    nested_full = create_full(size, fill_value)
    return Tensor(nested_full)


def eye(n):
    """Create an identity matrix of size n x n."""
    identity = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(Value(1.0 if i == j else 0.0))
        identity.append(row)
    return Tensor(identity)


def cat(tensors, dim=0):
    """Concatenate tensors along a dimension."""
    if dim == 0:
        concatenated = []
        for tensor in tensors:
            concatenated.extend(tensor.data)
        return Tensor(concatenated)
    elif dim == 1:
        concatenated = []
        for i, row in enumerate(tensors[0].data):
            concat_row = list(row)
            for tensor in tensors[1:]:
                concat_row.extend(tensor.data[i])
            concatenated.append(concat_row)
        return Tensor(concatenated)
    else:
        raise NotImplementedError(f"cat with dim={dim} not supported")


def stack(tensors, dim=0):
    """Stack tensors along a new dimension."""
    if dim == 0:
        stacked = [t.data for t in tensors]
        return Tensor(stacked)
    elif dim == 1:
        stacked = []
        for i, row in enumerate(tensors[0].data):
            stacked_row = [tensors[j].data[i] for j in range(len(tensors))]
            stacked.append(stacked_row)
        return Tensor(stacked)
    else:
        raise NotImplementedError(f"stack with dim={dim} not supported")


def clip_grad_norm_(parameters, max_norm):
    """Clip gradients by global norm (modifies gradients in-place)."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad ** 2
    
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
    
    return total_norm


def clip_grad_value_(parameters, clip_value):
    """Clip gradients by value (modifies gradients in-place)."""
    for p in parameters:
        if p.grad is not None:
            if p.grad > clip_value:
                p.grad = clip_value
            elif p.grad < -clip_value:
                p.grad = -clip_value


def one_hot(indices: Union[Tensor, List[int]], num_classes: int) -> Tensor:
    """Convert indices to one-hot encoding."""
    if isinstance(indices, Tensor):
        indices_list = indices._flatten(indices.data)
        indices_values = [int(v.data) if isinstance(v, Value) else int(v) for v in indices_list]
    else:
        indices_values = indices
    
    one_hot_data = []
    for idx in indices_values:
        row = [Value(1.0) if i == idx else Value(0.0) for i in range(num_classes)]
        one_hot_data.append(row)
    
    return Tensor(one_hot_data)


class DataLoader(object):
    """Simple DataLoader for batching data."""
    
    def __init__(self, data, batch_size=32, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data)))
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.data), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.data[idx] for idx in batch_indices]
            yield batch
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
