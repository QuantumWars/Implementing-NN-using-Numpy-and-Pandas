# Implementing-NN-using-Numpy-and-Pandas

![Image Description](https://cdn-images-1.medium.com/v2/resize:fit:1000/format:png/1*WJ57ZKta2HxlQhzxuWR5zw.png)
*Caption: https://pub.towardsai.net/implement-a-neural-network-from-scratch-with-numpy-67db290771b and https://commons.wikimedia.org/wiki/File:Multi-Layer_Neural_Network-Vector-Blank.svg*

This repository contains an implementation of a neural network from scratch using only NumPy, a fundamental library for numerical computing in Python. The neural network is designed to perform tasks such as classification, regression, or any other supervised learning problem.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Neural networks have shown remarkable capabilities in various machine learning tasks, and understanding their inner workings is crucial for mastering machine learning and deep learning concepts. This project serves as an educational resource and a practical implementation of a neural network using only NumPy.




## Getting Started

### Prerequisites

To run the code, you'll need:

- Python (>= 3.x)
- NumPy (>= 1.16)

### Installation

1. Clone the repository:

   ```bash
   
   git clone https://github.com/QuantumWars/Implementing-NN-using-Numpy-and-Pandas.git
   cd Implementing-NN-using-Numpy-and-Pandas
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv 
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
   OR 
    ```bash
   conda create -n ML 
   conda activate ML  
   ```
3. Install required packages:

   ```bash
   pip install numpy
   pip install pandas
   pip install matplotlib
   ```

## Data Preprocessing

The dataset is loaded from a CSV file using Pandas, and the data is normalized by dividing pixel values by 255. The dataset is split into training and development sets.



## Neural Network Architecture

The network has the following architecture:\
Input layer: 784 neurons (28x28 pixels) \
Hidden layer: 10 neurons \
Output layer: 10 neurons (one for each digit) \
Weights and biases are initialized randomly.

## Activation Functions
The ReLU (Rectified Linear Unit) function is used in the hidden layer, and the softmax function is used in the output layer.

## Forward Propagation
Forward propagation computes the activations of each layer by applying the weights, biases, and activation functions.

## Backward Propagation
Backward propagation calculates the gradients of the loss function with respect to each parameter. These gradients are used to update the parameters.

## Parameter Updates
Parameters are updated using gradient descent. The learning rate (alpha) controls the step size of the updates.

## Training the Network
The network is trained using the gradient descent algorithm, iterating over the training set and updating the parameters.

## Predictions
Predictions are made by performing forward propagation on the input data.

## Testing the Model
The model's performance can be tested on individual images from the training set.


## Conclusion
This project demonstrates the fundamental concepts of neural networks and how they can be implemented from scratch using NumPy and Pandas. While this implementation is basic and not optimized for performance, it provides a solid foundation for understanding the inner workings of neural networks.


