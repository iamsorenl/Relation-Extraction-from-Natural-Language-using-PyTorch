import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
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
        """
        Defines the forward pass of the MLP model.
        
        Parameters:
        - x: torch.Tensor, the input tensor with shape (batch_size, input_size)
        
        Returns:
        - torch.Tensor, the output tensor with shape (batch_size, output_size) containing probabilities in [0, 1]
        """

        # Step 1: Apply the first linear layer (input to hidden layer)
        # x initially contains the input features, with shape (batch_size, input_size)
        # The layer1 performs the operation: Z1 = W1 * X + b1
        # After this step, x will have shape (batch_size, hidden_size)
        x = self.layer1(x)

        # Step 2: Apply the ReLU activation function
        # The ReLU function is applied element-wise to x, setting all negative values to 0
        # ReLU introduces non-linearity, which helps the network learn complex patterns
        # After this step, x still has shape (batch_size, hidden_size), but with non-negative values
        x = self.relu(x)

        # Step 3: Apply the second linear layer (hidden layer to output layer)
        # This layer performs another linear transformation: Z2 = W2 * A1 + b2
        # This step maps the hidden layer activations to the output space (logits for each label)
        # After this step, x will have shape (batch_size, output_size)
        x = self.output(x)

        # Step 4: Apply the Sigmoid activation function
        # The Sigmoid function is applied to each element of the output vector
        # Sigmoid squashes the output values to the range [0, 1], converting the raw logits to probabilities
        # Each value in the final output represents the probability of a particular label being active
        # After this step, x still has shape (batch_size, output_size), but all values are in [0, 1]
        x = self.sigmoid(x)

        # Return the output tensor, containing probabilities for each label
        return x