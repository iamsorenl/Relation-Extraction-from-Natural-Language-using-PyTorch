import torch
import torch.nn as nn

class MLP(nn.Module):
    # Define constants within the class
    HIDDEN_SIZE = 128
    DROPOUT_PROB = 0.3

    def __init__(self, input_size, output_size):
        """
        Initialize the MLP model with input size and output size.
        Uses a fixed hidden size and dropout probability defined in the class constants.

        Parameters:
        - input_size: int, the size of the input vector (e.g., the number of features in your data)
        - output_size: int, the number of output labels (e.g., the number of relations to predict)
        """
        super(MLP, self).__init__()

        # First fully connected layer (input to hidden layer)
        self.layer1 = nn.Linear(input_size, self.HIDDEN_SIZE)

        # Leaky ReLU activation function with a negative slope
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(self.DROPOUT_PROB)

        # Second fully connected layer (hidden layer to output layer)
        self.output = nn.Linear(self.HIDDEN_SIZE, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the MLP model.
        
        Parameters:
        - x: torch.Tensor, the input tensor with shape (batch_size, input_size)
        
        Returns:
        - torch.Tensor, the output tensor with shape (batch_size, output_size)
        """

        # Step 1: Apply the first linear layer (input to hidden layer)
        x = self.layer1(x)

        # Step 2: Apply Leaky ReLU activation function
        x = self.leaky_relu(x)

        # Step 3: Apply Dropout during training
        x = self.dropout(x)

        # Step 4: Apply the second linear layer (hidden layer to output layer)
        x = self.output(x)

        # Return raw logits (without activation, needed for MultiLabelSoftMarginLoss)
        return x