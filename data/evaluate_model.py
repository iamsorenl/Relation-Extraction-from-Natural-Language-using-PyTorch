import torch

def evaluate_model(model, criterion, X_val, y_val):
    """
    Evaluate the model on the validation set.

    Parameters:
    - model: The MLP model.
    - criterion: The loss function (e.g., BCELoss).
    - X_val: Validation data (features as a PyTorch tensor).
    - y_val: Validation labels (as a PyTorch tensor).
    - device: The device (CPU or GPU) to run the model on.

    Returns:
    - val_loss: The calculated loss on the validation set.
    """
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():  # Disable gradient computation
        # Perform the forward pass to compute predictions
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)  # Calculate validation loss
    
    return val_loss.item()  # Return the validation loss value

def evaluate_accuracy(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the accuracy of the model on the test data.

    Parameters:
    - model: The trained MLP model.
    - X_test: Test data (features as a PyTorch tensor).
    - y_test: Test labels (as a PyTorch tensor).
    - device: The device (CPU or GPU) to run the model on.
    - threshold: Threshold to convert predicted probabilities to binary labels.

    Returns:
    - accuracy: The accuracy of the model on the test set.
    """
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():  # Disable gradient computation
        # Perform the forward pass to get predictions
        outputs = model(X_test)
        # Convert predicted probabilities to binary labels using the threshold
        predictions = (outputs >= threshold).float()
        # Calculate the number of correct predictions
        correct = (predictions == y_test).float().sum()
        # Calculate accuracy as the ratio of correct predictions to the total number of elements
        accuracy = correct / y_test.numel()
    
    return accuracy.item()  # Return the accuracy