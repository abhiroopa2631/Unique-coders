# Neural Network Regression README

This repository contains a Python script implementing a simple neural network for regression using NumPy. The neural network is trained to predict an output value based on input features. 

## Overview

The script `neural_network_regression.py` contains the implementation of the neural network. Here's a breakdown of the key components:

- **Normalization**: The input data `X` and output data `y` are normalized to a range between 0 and 1 using the maximum value of each feature.
- **Activation Function**: The sigmoid activation function is used both in the hidden layer and the output layer. 
- **Variable Initialization**: We randomly initialize the weights (`wh`, `wout`) and biases (`bh`, `bout`) of the neural network.
- **Training Loop**: The training loop runs for a specified number of epochs. In each epoch, it performs forward propagation to compute the output, calculates the error, and then performs backpropagation to update the weights.
- **Error Calculation**: The error is calculated as the difference between the predicted output and the actual output.

## Requirements

- Python 3.x
- NumPy

## Usage

To run the script, execute the following command:

python neural_network_regression.py
## Customization

- **Epochs**: You can modify the `epoch` variable to change the number of training epochs.
- **Learning Rate**: The learning rate `eta` determines the step size during weight updates. Adjust it according to your preference.
- **Network Architecture**: You can customize the number of neurons in the hidden layer (`hidden_neurons`) and the number of input and output neurons (`input_neurons`, `output_neurons`).

## Acknowledgments

This implementation is inspired by various resources and tutorials on neural networks and machine learning. 
   

