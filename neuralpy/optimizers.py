import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad


class RMSprop:
    def __init__(self, parameters, lr=0.001,
                 beta=0.9, eps=1e-8):

        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.cache = [
            np.zeros_like(param)
            for param, _ in parameters
        ]

    def step(self):
        for i, (param, grad) in enumerate(self.parameters):

            self.cache[i] = (
                self.beta * self.cache[i]
                + (1 - self.beta) * grad ** 2
            )

            param -= (
                self.lr * grad /
                (np.sqrt(self.cache[i]) + self.eps)
            )


class Adam:
    def __init__(self, parameters, lr=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8):

        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [
            np.zeros_like(param)
            for param, _ in parameters
        ]

        self.v = [
            np.zeros_like(param)
            for param, _ in parameters
        ]

        self.t = 0

    def step(self):
        self.t += 1

        for i, (param, grad) in enumerate(self.parameters):

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1 - self.beta1) * grad
            )

            self.v[i] = (
                self.beta2 * self.v[i]
                + (1 - self.beta2) * grad ** 2
            )

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= (
                self.lr * m_hat /
                (np.sqrt(v_hat) + self.eps)
            )