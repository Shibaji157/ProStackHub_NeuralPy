import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from neuralpy import Linear, ReLU, CrossEntropyLoss, Adam


# ---------------------------------------------------------
# NeuralPy - MNIST Training
# Neural network built from scratch using NumPy
# No PyTorch / TensorFlow
# ---------------------------------------------------------

np.random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


print("=" * 65)
print("          NeuralPy - MNIST Neural Network Training")
print("=" * 65)

# ---------------------------------------------------------
# 1. LOAD MNIST
# ---------------------------------------------------------

print("\n[1/7] Loading MNIST dataset...")
print("First run may take some time because MNIST is downloaded.")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

print("Dataset loaded successfully!")
print("Images :", X.shape)
print("Labels :", y.shape)


# ---------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------

print("\n[2/7] Preparing training and testing data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=10000,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# 3. NETWORK
# ---------------------------------------------------------

print("\n[3/7] Building NeuralPy network...")

layer1 = Linear(784, 256)
relu1 = ReLU()

layer2 = Linear(256, 128)
relu2 = ReLU()

layer3 = Linear(128, 10)

loss_function = CrossEntropyLoss()

parameters = (
    layer1.parameters()
    + layer2.parameters()
    + layer3.parameters()
)

optimizer = Adam(
    parameters,
    lr=0.001
)

print("Architecture:")
print("784 Input")
print("   ↓")
print("256 Linear + ReLU")
print("   ↓")
print("128 Linear + ReLU")
print("   ↓")
print("10 Output Classes")


# ---------------------------------------------------------
# 4. TRAINING SETTINGS
# ---------------------------------------------------------

epochs = 12
batch_size = 128

loss_history = []
accuracy_history = []

start_time = time.time()


# ---------------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------------

print("\n[4/7] Starting training...\n")

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))

    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0
    correct = 0
    total = 0

    batches = 0

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        xb = X_train[start:end]
        yb = y_train[start:end]

        # -----------------------
        # FORWARD
        # -----------------------

        z1 = layer1.forward(xb)
        a1 = relu1.forward(z1)

        z2 = layer2.forward(a1)
        a2 = relu2.forward(z2)

        logits = layer3.forward(a2)

        loss = loss_function.forward(
            logits,
            yb
        )

        # -----------------------
        # ACCURACY
        # -----------------------

        predictions = np.argmax(
            logits,
            axis=1
        )

        correct += np.sum(
            predictions == yb
        )

        total += len(yb)

        # -----------------------
        # BACKWARD
        # -----------------------

        grad = loss_function.backward()

        grad = layer3.backward(grad)

        grad = relu2.backward(grad)

        grad = layer2.backward(grad)

        grad = relu1.backward(grad)

        layer1.backward(grad)

        # -----------------------
        # OPTIMIZATION
        # -----------------------

        optimizer.step()

        epoch_loss += loss
        batches += 1

    avg_loss = epoch_loss / batches

    train_accuracy = (
        correct / total * 100
    )

    loss_history.append(avg_loss)
    accuracy_history.append(train_accuracy)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Training Accuracy: {train_accuracy:.2f}%"
    )


# ---------------------------------------------------------
# 6. TESTING
# ---------------------------------------------------------

print("\n[5/7] Evaluating test dataset...")

z1 = layer1.forward(X_test)
a1 = relu1.forward(z1)

z2 = layer2.forward(a1)
a2 = relu2.forward(z2)

test_logits = layer3.forward(a2)

test_predictions = np.argmax(
    test_logits,
    axis=1
)

test_accuracy = np.mean(
    test_predictions == y_test
) * 100

training_time = time.time() - start_time

print("\n" + "=" * 65)

print(
    f"FINAL TEST ACCURACY : "
    f"{test_accuracy:.2f}%"
)

print(
    f"TRAINING TIME       : "
    f"{training_time:.2f} seconds"
)

if test_accuracy >= 95:
    print(
        "STATUS              : "
        "✓ TARGET ACHIEVED (>95%)"
    )
else:
    print(
        "STATUS              : "
        "Target not yet reached"
    )

print("=" * 65)


# ---------------------------------------------------------
# 7. SAVE MODEL
# ---------------------------------------------------------

print("\n[6/7] Saving trained NeuralPy model...")

np.savez(
    "models/mnist_model.npz",

    layer1_weights=layer1.weights,
    layer1_bias=layer1.bias,

    layer2_weights=layer2.weights,
    layer2_bias=layer2.bias,

    layer3_weights=layer3.weights,
    layer3_bias=layer3.bias
)

print(
    "Model saved → "
    "models/mnist_model.npz"
)


# ---------------------------------------------------------
# VISUALIZATION 1
# LOSS CURVE
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker="o"
)

plt.title(
    "NeuralPy - MNIST Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/loss_curve.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# VISUALIZATION 2
# ACCURACY
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, epochs + 1),
    accuracy_history,
    marker="o"
)

plt.title(
    "NeuralPy - Training Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/accuracy_curve.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# VISUALIZATION 3
# WEIGHT DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    layer1.weights.flatten(),
    bins=50
)

plt.title(
    "NeuralPy - Layer 1 Weight Distribution"
)

plt.xlabel("Weight Value")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/weight_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# VISUALIZATION 4
# GRADIENT FLOW
# ---------------------------------------------------------

gradient_means = [
    np.mean(np.abs(layer1.dweights)),
    np.mean(np.abs(layer2.dweights)),
    np.mean(np.abs(layer3.dweights))
]

plt.figure(figsize=(8, 5))

plt.bar(
    ["Layer 1", "Layer 2", "Layer 3"],
    gradient_means
)

plt.title(
    "NeuralPy - Gradient Flow"
)

plt.xlabel("Layer")
plt.ylabel("Mean Absolute Gradient")

plt.tight_layout()

plt.savefig(
    "outputs/gradient_flow.png",
    dpi=200
)

plt.close()


print("\n[7/7] Visualizations generated!")

print(
    "✓ outputs/loss_curve.png"
)

print(
    "✓ outputs/accuracy_curve.png"
)

print(
    "✓ outputs/weight_distribution.png"
)

print(
    "✓ outputs/gradient_flow.png"
)

print("\n" + "=" * 65)
print("        NeuralPy TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)