import torch.nn as nn

class MLP(nn.Module):
    # Define constants within the class
    HIDDEN_SIZE = 128
    DROPOUT_PROB = 0.3

    def __init__(self, input_size, output_size):
        """
        Initialize the enhanced MLP model with input size and output size.
        - Two hidden layers with batch normalization and dropout.
        - Leaky ReLU as the activation function.

        Parameters:
        - input_size: int, the size of the input vector (e.g., the number of features in your data)
        - output_size: int, the number of output labels (e.g., the number of relations to predict)
        """
        super(MLP, self).__init__()

        # First fully connected layer (input to hidden layer 1)
        self.layer1 = nn.Linear(input_size, self.HIDDEN_SIZE)

        ###change here: Add batch normalization after the first hidden layer
        self.batch_norm1 = nn.BatchNorm1d(self.HIDDEN_SIZE)

        # Second fully connected layer (hidden layer 1 to hidden layer 2)
        self.layer2 = nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)

        ###change here: Add batch normalization after the second hidden layer
        self.batch_norm2 = nn.BatchNorm1d(self.HIDDEN_SIZE)

        # Leaky ReLU activation function with a negative slope
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        #self.leaky_relu = nn.ReLU()

        # Dropout layer for regularization (applied after each hidden layer)
        self.dropout = nn.Dropout(self.DROPOUT_PROB)

        # Output layer (hidden layer 2 to output layer)
        self.output = nn.Linear(self.HIDDEN_SIZE, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the enhanced MLP model.
        
        Parameters:
        - x: torch.Tensor, the input tensor with shape (batch_size, input_size)
        
        Returns:
        - torch.Tensor, the output tensor with shape (batch_size, output_size)
        """
        # Step 1: Apply the first linear layer (input to hidden layer 1)
        x = self.layer1(x)

        ###change here: Apply batch normalization and activation function
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)

        ###change here: Apply dropout for regularization
        x = self.dropout(x)

        # Step 2: Apply the second linear layer (hidden layer 1 to hidden layer 2)
        x = self.layer2(x)

        ###change here: Apply batch normalization and activation function
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)

        ###change here: Apply dropout for regularization
        x = self.dropout(x)

        # Step 3: Apply the output layer (hidden layer 2 to output)
        x = self.output(x)

        # Return raw logits (without activation, needed for MultiLabelSoftMarginLoss)
        return x