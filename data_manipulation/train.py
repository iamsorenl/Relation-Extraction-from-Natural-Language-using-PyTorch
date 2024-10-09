import torch

def train_one_epoch(model, criterion, optimizer, X_train, y_train, device):
    """
    Train the model for one epoch.

    Parameters:
    - model: The MLP model.
    - criterion: The loss function (e.g., BCELoss).
    - optimizer: The optimizer (e.g., Adam).
    - X_train: Training data (features as a PyTorch tensor).
    - y_train: Training labels (as a PyTorch tensor).
    - device: The device (CPU or GPU) to run the model on.

    Returns:
    - loss_value: The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    
    # Move data to the specified device (CPU or GPU)
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    # Perform the forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Perform the backward pass and optimization step
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Backpropagate the loss
    optimizer.step()       # Update the model parameters
    
    return loss.item()  # Return the loss value

def evaluate_model(model, criterion, X_val, y_val, device):
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
    
    # Move data to the specified device (CPU or GPU)
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    with torch.no_grad():  # Disable gradient computation
        # Perform the forward pass to compute predictions
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)  # Calculate validation loss
    
    return val_loss.item()  # Return the validation loss value

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, device, num_epochs=50):
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
    - device: The device (CPU or GPU) to run the model on.
    - num_epochs: int, number of epochs to train for.

    Returns:
    - model: The trained model.
    """
    for epoch in range(num_epochs):
        # Train the model for one epoch
        loss = train_one_epoch(model, criterion, optimizer, X_train, y_train, device)
        
        # Evaluate the model on the validation set
        val_loss = evaluate_model(model, criterion, X_val, y_val, device)
        
        # Print the training and validation losses for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    return model  # Return the trained model

def evaluate_accuracy(model, X_test, y_test, device, threshold=0.5):
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
    
    # Move data to the specified device (CPU or GPU)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
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

def save_model(model, path='trained_model.pth'):
    """
    Save the trained model to a file.

    Parameters:
    - model: The trained MLP model.
    - path: str, the file path to save the model.
    """
    # Save the model's state dictionary to the specified path
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")