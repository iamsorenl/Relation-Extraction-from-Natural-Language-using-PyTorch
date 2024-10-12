import torch
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, batch_size, mlb):
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

        # Evaluate the model on the validation set
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():  # Disable gradient computation
            for X_batch, y_batch in val_loader:
                # Forward pass
                val_outputs = model(X_batch)
                
                # Apply sigmoid to convert logits to probabilities
                val_probabilities = torch.sigmoid(val_outputs)
                
                # Calculate the validation loss
                val_loss = criterion(val_outputs, y_batch)
                running_val_loss += val_loss.item()

                # Perform 1-to-1 accuracy check
                val_predictions = (val_probabilities >= 0.5).float()
                total_samples += y_batch.size(0)

                # Inverse transform predictions and true labels to original label sets
                predicted_labels = mlb.inverse_transform(val_predictions.cpu().numpy())
                true_labels = mlb.inverse_transform(y_batch.cpu().numpy())

                # Calculate the number of correct predictions for 1-to-1 matching
                correct_predictions += sum(set(predicted_labels[i]) == set(true_labels[i]) for i in range(len(true_labels)))

        # Calculate average validation loss
        avg_val_loss = running_val_loss / len(val_loader)
        
        # Calculate 1-to-1 comparison accuracy
        comparison_accuracy = correct_predictions / total_samples

        # Print the training and validation losses, and 1-to-1 accuracy for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, 1-to-1 Comparison Accuracy: {comparison_accuracy:.4f}")
    
    return model