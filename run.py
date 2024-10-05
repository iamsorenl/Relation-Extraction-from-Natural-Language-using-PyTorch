import sys
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# define hyperparameter
hidden_size = 128

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        """
        Initialize the MLP model with input size, output size, and an adjustable hidden layer size.

        Parameters:
        - input_size: int, the size of the input vector (e.g., the number of features in your data)
        - output_size: int, the number of output labels (e.g., the number of relations to predict)
        - hidden_size: int, the number of neurons in the hidden layer (default: 128)
        """
        super(MLP, self).__init__()

        # First fully connected layer (input to hidden layer)
        # This layer performs a linear transformation: Z1 = W1 * X + b1
        # Input: a vector of size (input_size,)
        # Output: a vector of size (hidden_size,)
        self.layer1 = nn.Linear(input_size, hidden_size)

        # Activation function: ReLU (Rectified Linear Unit)
        # This function is applied element-wise to the output of the first layer
        # It introduces non-linearity by outputting max(0, Z1), setting negative values to 0
        self.relu = nn.ReLU()

        # Second fully connected layer (hidden layer to output layer)
        # This layer performs another linear transformation: Z2 = W2 * A1 + b2
        # Input: a vector of size (hidden_size,)
        # Output: a vector of size (output_size,)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation function: Sigmoid
        # This is applied to the final output, converting the logits to probabilities
        # Sigmoid squashes the output values to the range [0, 1], making them interpretable as probabilities
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Define the forward pass here
        return x