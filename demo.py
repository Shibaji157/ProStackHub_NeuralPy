import numpy as np

from neuralpy import (
    Tensor,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    MSELoss,
    CrossEntropyLoss,
    SGD,
    Adam,
    RMSprop
)


print("=" * 55)
print("       NeuralPy - Neural Network Library")
print("=" * 55)

# Tensor test
x = Tensor(
    [2.0, 3.0],
    requires_grad=True
)

y = x * 2

print("\nTensor Test")
print("Input :", x)
print("Output:", y)

# Linear Layer test
layer = Linear(4, 3)

sample = np.random.randn(
    2,
    4
).astype(np.float32)

output = layer.forward(sample)

print("\nLinear Layer Test")
print("Input Shape :", sample.shape)
print("Output Shape:", output.shape)

# Activation tests
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()

test = np.array([
    [-2, -1, 0, 1, 2]
], dtype=np.float32)

print("\nActivation Function Tests")

print(
    "ReLU:",
    relu.forward(test)
)

print(
    "Sigmoid:",
    sigmoid.forward(test)
)

print(
    "Tanh:",
    tanh.forward(test)
)

print("\nAvailable Optimizers:")
print("✓ SGD")
print("✓ Adam")
print("✓ RMSprop")

print("\nAvailable Loss Functions:")
print("✓ Mean Squared Error")
print("✓ Cross Entropy")

print("\nNeuralPy successfully initialized!")