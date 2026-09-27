import numpy as np


class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.input > 0)


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -50, 50)
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)


class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1.0 - self.output ** 2)