import numpy as np


class MSELoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        return np.mean((predictions - targets) ** 2)

    def backward(self):
        return (
            2.0 * (self.predictions - self.targets)
            / self.targets.size
        )


class CrossEntropyLoss:
    def forward(self, logits, targets):
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        exp_values = np.exp(shifted)

        self.probabilities = (
            exp_values /
            np.sum(exp_values, axis=1, keepdims=True)
        )

        self.targets = targets
        n = logits.shape[0]

        correct_probs = self.probabilities[
            np.arange(n), targets
        ]

        return -np.mean(np.log(correct_probs + 1e-12))

    def backward(self):
        n = self.probabilities.shape[0]

        grad = self.probabilities.copy()
        grad[np.arange(n), self.targets] -= 1

        return grad / n