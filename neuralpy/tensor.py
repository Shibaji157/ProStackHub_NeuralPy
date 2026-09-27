import numpy as np


class Tensor:
    """
    Basic Tensor class with automatic differentiation support.
    """

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad

        if requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        self._backward = lambda: None
        self._prev = set()

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad

            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad

            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return (
            f"Tensor(data={self.data}, "
            f"requires_grad={self.requires_grad})"
        )