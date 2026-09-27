import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


print("=" * 60)
print("        NeuralPy - ONNX Model Export")
print("=" * 60)

model_path = "models/mnist_model.npz"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Trained model not found. Run train_mnist.py first."
    )

print("\n[1/4] Loading NeuralPy weights...")

data = np.load(model_path)

W1 = data["layer1_weights"].astype(np.float32)
b1 = data["layer1_bias"].reshape(-1).astype(np.float32)

W2 = data["layer2_weights"].astype(np.float32)
b2 = data["layer2_bias"].reshape(-1).astype(np.float32)

W3 = data["layer3_weights"].astype(np.float32)
b3 = data["layer3_bias"].reshape(-1).astype(np.float32)

print("✓ Weights loaded successfully")


print("\n[2/4] Building ONNX graph...")

input_tensor = helper.make_tensor_value_info(
    "input",
    TensorProto.FLOAT,
    [None, 784]
)

output_tensor = helper.make_tensor_value_info(
    "output",
    TensorProto.FLOAT,
    [None, 10]
)

initializers = [
    numpy_helper.from_array(W1, name="W1"),
    numpy_helper.from_array(b1, name="b1"),

    numpy_helper.from_array(W2, name="W2"),
    numpy_helper.from_array(b2, name="b2"),

    numpy_helper.from_array(W3, name="W3"),
    numpy_helper.from_array(b3, name="b3")
]

nodes = [
    helper.make_node(
        "MatMul",
        ["input", "W1"],
        ["mm1"]
    ),

    helper.make_node(
        "Add",
        ["mm1", "b1"],
        ["z1"]
    ),

    helper.make_node(
        "Relu",
        ["z1"],
        ["a1"]
    ),

    helper.make_node(
        "MatMul",
        ["a1", "W2"],
        ["mm2"]
    ),

    helper.make_node(
        "Add",
        ["mm2", "b2"],
        ["z2"]
    ),

    helper.make_node(
        "Relu",
        ["z2"],
        ["a2"]
    ),

    helper.make_node(
        "MatMul",
        ["a2", "W3"],
        ["mm3"]
    ),

    helper.make_node(
        "Add",
        ["mm3", "b3"],
        ["output"]
    )
]

graph = helper.make_graph(
    nodes,
    "NeuralPyMNIST",
    [input_tensor],
    [output_tensor],
    initializer=initializers
)

model = helper.make_model(
    graph,
    producer_name="NeuralPy"
)

model.opset_import[0].version = 13

print("✓ ONNX graph created")


print("\n[3/4] Validating model...")

onnx.checker.check_model(model)

print("✓ ONNX validation successful")


print("\n[4/4] Saving model...")

output_path = "models/neuralpy_mnist.onnx"

onnx.save(
    model,
    output_path
)

print(f"✓ Model saved → {output_path}")

print("\n" + "=" * 60)
print("       ONNX EXPORT COMPLETED SUCCESSFULLY")
print("=" * 60)