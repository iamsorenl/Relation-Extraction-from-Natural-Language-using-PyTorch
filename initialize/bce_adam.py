import torch
import torch.nn as nn
from models.mlp import MLP

def initialize_mlp_bce_adam(input_size, output_size, hidden_size=128, learning_rate=0.001):
    """
    Initialize the MLP model, loss function (Binary Cross-Entropy Loss), and Adam optimizer.

    Parameters:
    - input_size: int, the number of input features.
    - output_size: int, the number of output classes (relations).
    - hidden_size: int, the number of neurons in the hidden layer.
    - learning_rate: float, learning rate for the optimizer.

    Returns:
    - model: The initialized MLP model.
    - criterion: Loss function (Binary Cross-Entropy Loss).
    - optimizer: Optimizer (Adam).
    """
    # Initialize the model
    model = MLP(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    
    # Loss function (Binary Cross-Entropy for multi-label classification)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    
    # Optimizer (Adam optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer