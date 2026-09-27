import numpy as np


class Linear:
    def __init__(self, input_size, output_size):

        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))

        self.weights = np.random.uniform(
            -limit,
            limit,
            (input_size, output_size)
        ).astype(np.float32)

        self.bias = np.zeros(
            (1, output_size),
            dtype=np.float32
        )

        # IMPORTANT:
        # These arrays must remain the same objects
        # because the optimizer keeps references to them.
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

        self.input = None

    def forward(self, x):

        self.input = x

        return (
            np.dot(x, self.weights)
            + self.bias
        )

    def backward(self, grad_output):

        # IMPORTANT FIX:
        # Update gradient arrays IN PLACE.
        # Do not replace self.dweights/self.dbias.

        self.dweights[:] = np.dot(
            self.input.T,
            grad_output
        )

        self.dbias[:] = np.sum(
            grad_output,
            axis=0,
            keepdims=True
        )

        grad_input = np.dot(
            grad_output,
            self.weights.T
        )

        return grad_input

    def parameters(self):

        return [
            (self.weights, self.dweights),
            (self.bias, self.dbias)
        ]