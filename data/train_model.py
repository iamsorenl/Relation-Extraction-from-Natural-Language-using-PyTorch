from torch.utils.data import DataLoader, TensorDataset
from data.evaluate_model import evaluate_model

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, batch_size):
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

    # Create DataLoader for training and validation sets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0.0

        # Mini-batch training
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)

        # Evaluate the model on the validation set using the evaluate_model function
        val_loss = evaluate_model(model, criterion, X_val, y_val)
        
        # Print the training and validation losses for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    return model