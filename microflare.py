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
