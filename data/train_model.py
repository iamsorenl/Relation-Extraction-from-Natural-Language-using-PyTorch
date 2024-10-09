import torch
from data.evaluate_model import evaluate_model

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs=50):
    """
    Train the model over multiple epochs, evaluating on the validation set after each epoch.

    Parameters:
    - model: The MLP model.
    - criterion: The loss function (e.g., BCELoss).
    - optimizer: The optimizer (e.g., Adam).
    - X_train: Training data (features as a PyTorch tensor).
    - y_train: Training labels (as a PyTorch tensor).
    - X_val: Validation data (features as a PyTorch tensor).
    - y_val: Validation labels (as a PyTorch tensor).
    - num_epochs: int, number of epochs to train for.

    Returns:
    - model: The trained model.
    """
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Forward pass for training data
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the model on the validation set using the evaluate_model function
        val_loss = evaluate_model(model, criterion, X_val, y_val)
        
        # Print the training and validation losses for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    return model