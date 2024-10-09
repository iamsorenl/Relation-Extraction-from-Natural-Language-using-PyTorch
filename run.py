import sys
import torch
from models.mlp import MLP
from data.load_data import load_data, split_data
from data.preprocess import preprocess_data
from data.submission import create_submission_file
from data.train import train_model, evaluate_model, evaluate_accuracy, save_model
from initialize.bce_adam import initialize_mlp_bce_adam

def main(train_file, test_file, num_epochs=500, hidden_size=128, learning_rate=0.001):
    """
    Main function to run the training pipeline.

    Parameters:
    - train_file: str, path to the training data CSV file.
    - test_file: str, path to the test data CSV file.
    - num_epochs: int, the number of epochs to train the model.
    - hidden_size: int, the number of neurons in the hidden layer.
    - learning_rate: float, the learning rate for the optimizer.
    """
    # Select the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Load and split the data
    train_df = load_data(train_file, test_file)
    train_set, val_set, test_set = split_data(train_df)

    # Step 2: Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test, mlb = preprocess_data(train_set, val_set, test_set)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Step 3: Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model, criterion, optimizer = initialize_mlp_bce_adam(input_size, output_size, hidden_size, learning_rate)

    # Move the model to the selected device
    model.to(device)

    # Step 4: Train the model
    model = train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, device, num_epochs)

    # Step 5: Evaluate the model on the test set
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    test_loss = evaluate_model(model, criterion, X_test_tensor, y_test_tensor, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Step 6: Evaluate accuracy on the test set
    test_accuracy = evaluate_accuracy(model, X_test_tensor, y_test_tensor, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Step 7: Create a submission file
    create_submission_file(model, X_test_tensor, mlb, output_file='submission.csv')

    # Step 8: Save the trained model
    save_model(model, 'trained_model.pth')

if __name__ == "__main__":
    # Example usage: python run.py hw1_train.csv hw1_test.csv
    if len(sys.argv) != 3:
        print("Usage: python run.py <train_data> <test_data>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)