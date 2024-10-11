import sys
import torch
import torch.nn as nn
import pandas as pd
from models.mlp import MLP
from data.load_split_data import load_and_split_data
from data.preprocess import preprocess_data
from data.train_model import train_model
from data.evaluate_model import evaluate_model, evaluate_accuracy

def main(train_file, test_file, num_epochs=350, hidden_size=128, learning_rate=0.001):
    """
    Main function to run the training pipeline.

    Parameters:
    - train_file: str, path to the training data CSV file.
    - test_file: str, path to the test data CSV file.
    - num_epochs: int, the number of epochs to train the model.
    - hidden_size: int, the number of neurons in the hidden layer.
    - learning_rate: float, the learning rate for the optimizer.
    """

    # Step 1: Load and split the training data into training, validation, and test sets
    # This function loads the training data and splits it into three sets:
    # - train_set: For training the model
    # - val_set: For validation during training
    # - test_set: Used to evaluate the model after training
    train_set, val_set, test_set = load_and_split_data(train_file)

    # Step 2: Preprocess the data
    # Converts the text data into a Bag-of-Words representation and the labels into binary form
    # Returns the processed training, validation, and test data along with the fitted vectorizer and MultiLabelBinarizer
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb = preprocess_data(train_set, val_set, test_set)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Step 3: Initialize the model, loss function, and optimizer
    # The model, loss function (BCELoss), and optimizer (Adam) are created directly here
    input_size = X_train.shape[1]  # Number of input features
    output_size = y_train.shape[1]  # Number of output labels
    model = MLP(input_size, output_size, hidden_size)  # Initialize the MLP model
    criterion = nn.BCELoss()  # Use Binary Cross-Entropy Loss for multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Step 4: Train the model using the training and validation data
    # The training function will train the model for the specified number of epochs, and monitor validation loss
    trained_model = train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs)

    # Step 5: Evaluate the trained model on the test set (split from training data)
    # Evaluate the model's performance using the test set and print the resulting loss
    test_loss = evaluate_model(trained_model, criterion, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Step 6: Evaluate accuracy on the test set
    # Calculate and print the accuracy of the model on the test set
    test_accuracy = evaluate_accuracy(trained_model, X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Step 7: Preprocess the actual test file (from the test_file parameter) for submission
    # The test file contains the actual data to make predictions on for creating a submission
    test_df = pd.read_csv(test_file)  # Load the test file into a DataFrame
    X_test_submission = vectorizer.transform(test_df['UTTERANCES'])  # Use the fitted vectorizer to transform the data
    X_test_submission_tensor = torch.tensor(X_test_submission.toarray(), dtype=torch.float32)  # Convert to tensor

    # Step 8: Generate predictions for the test file and create a submission file
    # Perform predictions on the test file data and create a DataFrame for submission
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(X_test_submission_tensor)
        predictions = (outputs >= 0.5).float()
        predicted_labels = mlb.inverse_transform(predictions.cpu().numpy())

    # Create the submission DataFrame
    submission = pd.DataFrame({
        'ID': range(len(predicted_labels)),
        'Core Relations': [' '.join(sorted(labels)) if labels else '' for labels in predicted_labels]
    })
    
    # Save the submission DataFrame to a CSV file
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved to submission.csv")

if __name__ == "__main__":
    # Example usage: python run.py hw1_train.csv hw1_test.csv
    if len(sys.argv) != 3:
        print("Usage: python run.py <train_data> <test_data>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)