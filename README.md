

## 🌐 Live Demo

🚀 **Live Application:** https://pro-stack-hub-neural-py.vercel.app

💻 **GitHub Repository:** https://github.com/Shibaji157/ProStackHub_NeuralPy

---

## 🏆 Key Result

NeuralPy achieved **97.97% test accuracy on MNIST**, exceeding the project target of 95%.

The neural network was implemented from scratch using Python and NumPy without PyTorch or TensorFlow.

# 🧠 NeuralPy — Neural Network Library From Scratch

NeuralPy is a lightweight neural network framework implemented from scratch using Python and NumPy.

The project demonstrates the internal working of neural networks without relying on deep-learning frameworks such as TensorFlow or PyTorch.

## 🎯 Project Objective

The objective of NeuralPy is to understand and implement the fundamental components of a neural-network framework, including tensors, layers, activation functions, loss functions, optimizers, forward propagation, backpropagation, model training, visualization, and model export.

## ✨ Features

- Custom Tensor class
- Basic automatic differentiation
- Linear/Fully Connected layers
- ReLU activation
- Sigmoid activation
- Tanh activation
- Mean Squared Error Loss
- Cross Entropy Loss
- SGD optimizer
- RMSprop optimizer
- Adam optimizer
- Forward propagation
- Backpropagation
- MNIST digit classification
- Training-loss visualization
- Training-accuracy visualization
- Weight-distribution analysis
- Gradient-flow visualization
- NumPy model persistence
- ONNX model export

## 🏗️ Neural Network Architecture

```text
Input Layer
784 Features
     │
     ▼
Linear Layer
784 → 256
     │
     ▼
ReLU
     │
     ▼
Linear Layer
256 → 128
     │
     ▼
ReLU
     │
     ▼
Output Layer
128 → 10
```

## 📊 Dataset

The project uses the MNIST handwritten-digit dataset.

- Total samples: 70,000
- Training samples: 60,000
- Testing samples: 10,000
- Input features: 784
- Output classes: 10

## 🏆 Result

The NeuralPy model achieved:

**Final Test Accuracy: 97.97%**

This exceeds the project target of 95% test accuracy.

## 📈 Training Visualizations

The project automatically generates:

- `loss_curve.png`
- `accuracy_curve.png`
- `weight_distribution.png`
- `gradient_flow.png`

These visualizations help analyze model convergence, performance, learned parameters, and gradient behavior.

## 💾 Model Export

Trained parameters are stored in:

```text
models/mnist_model.npz
```

The trained architecture and parameters are also exported to:

```text
models/neuralpy_mnist.onnx
```

for interoperability with ONNX-compatible tools.

## 📁 Project Structure

```text
NeuralPy/
│
├── neuralpy/
│   ├── __init__.py
│   ├── tensor.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   └── optimizers.py
│
├── models/
│   ├── mnist_model.npz
│   └── neuralpy_mnist.onnx
│
├── outputs/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── gradient_flow.png
│   └── weight_distribution.png
│
├── demo.py
├── train_mnist.py
├── export_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run NeuralPy Demo

```bash
python demo.py
```

## 🧠 Train on MNIST

```bash
python train_mnist.py
```

The trained model and visualization outputs will be generated automatically.

## 📦 Export to ONNX

```bash
python export_model.py
```

## 🛠️ Technologies

- Python
- NumPy
- Matplotlib
- Scikit-learn
- ONNX

## 📚 Concepts Demonstrated

This project demonstrates:

- Neural networks from first principles
- Matrix-based forward propagation
- Backpropagation
- Gradient descent
- Automatic differentiation fundamentals
- Loss optimization
- Weight initialization
- Model evaluation
- Gradient-flow analysis
- Model serialization and interoperability

## 👨‍💻 Internship Project

Developed as part of the **ProStackHub Python Programming Internship**.

**Project:** NeuralPy — A From-Scratch Neural Network Library

**Developer:** Shibaji Biswas
**Email :** shibajibiswas.cse@gmail.com
       Chandigarh University