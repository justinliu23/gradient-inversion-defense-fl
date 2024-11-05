# gradient-inversion-defense-fl

This repository contains an implementation of a gradient inversion attack simulation, where different defense mechanisms are applied to evaluate their effectiveness against data reconstruction attempts. The attack targets a Convolutional Neural Network (CNN) based on the LeNet architecture, using gradients from the CIFAR-100 dataset to reconstruct input data. The experiment tests several defense strategies to obscure gradients, assessing the robustness of these techniques in preserving data privacy during collaborative or federated learning.

## Project Overview

In collaborative learning settings, gradient-based attacks allow adversaries to reconstruct input data using gradients shared by participating clients. This project explores:
1. **Gradient-Based Data Reconstruction**: Simulating an attack to reconstruct original data (images) from shared gradients.
2. **Defense Mechanisms**: Applying various gradient defense strategies to reduce the risk of data leakage.
3. **Defense Effectiveness**: Observing and comparing how each defense affects the reconstructed data quality over multiple iterations.

### Core Concepts

- **Gradient Inversion Attack**: Attempts to reconstruct the input data by matching the gradients of random data (dummy data) to those of the original data.
- **Defense Strategies**: Methods applied to gradients to protect data, including pruning, quantization, and noise injection.
- **LeNet Model**: A simple CNN used to classify images in CIFAR-100, targeted by the attack.

## Key Components

### 1. Model and Dataset Preparation
- **Dataset**: The CIFAR-100 dataset is used, with images preprocessed to 32x32 resolution.
- **Model**: A basic LeNet CNN with Sigmoid activations is employed for classification, containing four convolutional layers and one fully connected layer.

### 2. Defense Mechanisms
- **None**: No defense; gradients are fully exposed.
- **Pruning**: Removes the smallest 20% of gradient values by setting them to zero, thereby reducing gradient details.
- **Quantization**: Simulates quantization to a lower precision (4 bits) by rounding gradient values, limiting the information each value represents.
- **Noise Injection**: Adds Gaussian noise to gradients, reducing their accuracy and potentially obscuring fine details that could aid reconstruction.

The code tests each defense strategy separately, applying it to gradients before attempting data reconstruction.

### 3. Attack Simulation
- **Dummy Data Generation**: Dummy data (randomly initialized images and labels) are iteratively optimized to match the defended gradients of the original data.
- **Optimization Process**: An LBFGS optimizer minimizes the difference between gradients of dummy data and the defended gradients, gradually making the dummy data resemble the original.
- **Loss Calculation**: The loss is defined by the gradient difference between the dummy data and defended gradients, with the optimizer backpropagating this difference to update the dummy data.

### 4. Visualization and Evaluation
- **Progression Tracking**: Every 10 iterations, the current state of the dummy data is recorded to observe the reconstruction progress.
- **Result Plots**: Each defense strategy’s effectiveness is visualized in separate plots, showing the evolution of the dummy data over time and comparing the reconstructed image quality across defense types.

## Code Structure and Functions

### Primary Functions
- **label_to_onehot**: Converts labels to a one-hot encoding format for classification tasks.
- **cross_entropy_for_onehot**: Computes cross-entropy loss for one-hot encoded labels, used for gradient calculation.
- **defense_method**: Applies the specified defense strategy (`none`, `pruning`, `quantization`, or `noise`) to the gradients.

### Attack Loop
- For each defense strategy:
  1. **Gradient Defense Application**: Modifies the original gradients according to the chosen defense strategy.
  2. **Dummy Data Optimization**: Uses LBFGS to update dummy data until its gradients approximate the defended gradients.
  3. **Progress Visualization**: Records and visualizes reconstruction progress every 10 iterations.

## Requirements

- **Python 3.6+**
- **Libraries**:
  - `torch`: For model definition, gradient calculation, and optimization.
  - `torchvision`: For dataset loading and transformations.
  - `matplotlib`: For visualizing reconstructed images.
