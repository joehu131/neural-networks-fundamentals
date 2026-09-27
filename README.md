# Machine Learning & Neural Networks

A structured collection of machine learning algorithms, deep neural network architectures, ensemble methods, and reinforcement learning systems. Implementations range from first-principles vector calculus in NumPy to deep convolutional networks in TensorFlow/Keras.

---

## Technical Overview

```
                        ┌────────────────────────────────────────────────────────┐
                        │      Machine Learning Implementation Roadmap           │
                        └──────────────────────────┬─────────────────────────────┘
                                                   │
         ┌───────────────────┬─────────────────────┼─────────────────────┬───────────────────┐
         │                   │                     │                     │                   │
         ▼                   ▼                     ▼                     ▼                   ▼
┌─────────────────┐ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ ┌─────────────────┐
│ Classical ML    │ │ Perceptrons &   │   │ Deep Learning   │   │ Ensemble        │ │ Reinforcement   │
│                 │ │ Backpropagation │   │ (CNNs)          │   │ Learning        │ │ Learning        │
│ • k-NN          │ │ • Single-layer  │   │ • MNIST Demo    │   │ • AdaBoost      │ │ • Q-Learning    │
│ • Cross-Valid.  │ │ • Multi-layer   │   │ • CIFAR-10 Lab  │   │ • Haar Features │ │ • GridWorld     │
│ • Metrics/CM    │ │ • Pure NumPy    │   │ • Keras/TF      │   │ • Face Detect   │ │ • RocketWorld   │
└─────────────────┘ └─────────────────┘   └─────────────────┘   └─────────────────┘ └─────────────────┘
```

---

## Modules

### 1. Nearest Neighbor Classification (`01-classical-knn`)
- **Algorithms**: k-NN [k-Nearest Neighbors] classification with Euclidean distance metric.
- **Evaluation**: k-fold cross-validation for hyperparameter tuning ($k \in [1, 50]$), confusion matrix generation, multi-class accuracy calculation.
- **Key Concepts**: Decision boundary smoothness versus overfitting, curse of dimensionality, non-parametric estimation.

### 2. Perceptrons & Backpropagation from Scratch (`02-perceptrons-backpropagation`)
- **Algorithms**:
  - Single-Layer Perceptron: Linear classification, Mean Squared Error loss, delta learning rule.
  - Multi-Layer Perceptron (MLP): Forward propagation, analytical gradient calculation, backpropagation with non-linear activations ($\tanh$, sigmoid), mini-batch gradient descent.
- **Key Concepts**: Linear separability limitations, solving XOR and concentric ring topologies, decision surface visualization.

### 3. Deep Convolutional Neural Networks (`03-deep-learning-cnn`)
- **Architectures**:
  - Baseline feedforward networks on MNIST digit classification.
  - Custom deep CNNs [Convolutional Neural Networks] for 10-class object recognition on CIFAR-10.
- **Techniques**: Convolutional filter design, max pooling, dropout regularization, batch normalization, learning rate scheduling, confusion matrix analysis per class.
- **Framework**: TensorFlow 2 / Keras.

### 4. Ensemble Learning: AdaBoost & Facial Detection (`04-adaboost-face-detection`)
- **Algorithms**:
  - AdaBoost ensemble optimization with decision stumps as weak classifiers.
  - Viola-Jones style Haar-like feature extraction (two-rectangle and three-rectangle masks).
- **Application**: Face versus non-face binary classification, followed by full sliding-window detection and heatmap localization on historical group portraits (such as the 1927 Solvay Conference).
- **Implementation**: Pure vectorized NumPy (precomputed feature-response matrices, exponential sample re-weighting, threshold search).

### 5. Reinforcement Learning: Temporal Difference Q-Learning (`05-reinforcement-learning-qlearning`)
- **Algorithms**: Tabular Q-Learning (model-free temporal difference control).
- **Environments**:
  - `GridWorld`: Obstacle navigation with negative step rewards, goal terminal states, and cliff avoidance.
  - `RocketWorld`: Dynamic lander control simulation.
- **Key Concepts**: Bellman optimality equation, exploration versus exploitation ($\epsilon$-greedy decay), discount factor ($\gamma$) dynamics, state-value function $V(s) = \max_a Q(s, a)$ visualization.

---

## Getting Started

### Prerequisites
- Python 3.9+ or Anaconda / Miniconda

### Installation

```bash
# Clone the repository
git clone https://github.com/joehu131/neural-networks-fundamentals.git
cd neural-networks-fundamentals

# Create and activate environment using Conda
conda env create -f environment.yml
conda activate ml-portfolio

# Alternatively, install using pip
pip install -r requirements.txt
```

### Running the Notebooks

Launch JupyterLab to interact with any module:

```bash
jupyter lab
```

Recommended execution order:
1. `01-classical-knn/kNN.ipynb`
2. `02-perceptrons-backpropagation/SingleLayer.ipynb` followed by `MultiLayer.ipynb`
3. `03-deep-learning-cnn/CIFAR10-Lab.ipynb`
4. `04-adaboost-face-detection/AdaBoost.ipynb`
5. `05-reinforcement-learning-qlearning/QLearning.ipynb`

---

## Repository Structure

```
NeuralNetworks/
├── 01-classical-knn/
│   ├── Data/                    # Synthetic 2D classification datasets
│   ├── NotebookMaterial/        # Cross-validation illustration assets
│   ├── evalFunctions.py         # Confusion matrix and accuracy metrics
│   ├── kNN.ipynb                # Main k-NN experiment notebook
│   └── utils.py                 # Plotting and synthetic dataset generation
├── 02-perceptrons-backpropagation/
│   ├── Data/                    # Classification datasets
│   ├── SingleLayer.ipynb        # Single-layer perceptron experiments
│   ├── TwoLayer.ipynb           # Two-layer network with backprop
│   ├── MultiLayer.ipynb         # General deep MLP implementation
│   ├── evalFunctions.py         # Performance evaluation metrics
│   └── utils.py                 # Decision boundary plotting utilities
├── 03-deep-learning-cnn/
│   ├── CIFAR10-Lab.ipynb        # Deep CNN architecture and benchmark
│   ├── MNIST-Demo.ipynb         # Baseline digit classification
│   └── Custom.py                # History and confusion matrix visualizers
├── 04-adaboost-face-detection/
│   ├── Data/                    # Face, non-face, and Solvay test images
│   ├── NotebookMaterials/       # Haar filter and decision stump schematics
│   ├── AdaBoost.ipynb           # AdaBoost implementation and face detection
│   └── utils.py                 # Haar feature mask generator and visualizer
├── 05-reinforcement-learning-qlearning/
│   ├── QLearning.ipynb          # Q-Learning algorithm implementation
│   ├── gridworld.py             # GridWorld environments (worlds 1-8)
│   ├── rocketworld.py           # Rocket continuous control environment
│   ├── world.py                 # Base environment class
│   └── utils.py                 # Policy and value extraction routines
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip dependency list
└── README.md                    # Project documentation
```
