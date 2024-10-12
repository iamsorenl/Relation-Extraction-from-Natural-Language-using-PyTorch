import sys
import torch
import torch.nn as nn
import pandas as pd
from models.mlp import MLP
import gensim.downloader as api
from data.load_split_data import load_and_split_data
from data.preprocess import preprocess_data, convert_text_to_glove_embeddings  # Import the function
from data.train_model import train_model
from data.evaluate_model import evaluate_model, evaluate_accuracy
import numpy as np

# Ensure nltk's punkt tokenizer is available
import nltk
nltk.download('punkt_tab')

def main(train_file, test_file, num_epochs=35, hidden_size=128, dropout_prob=0.1, learning_rate=0.001, batch_size=32):
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

    # Step 2: Load GloVe embeddings
    wv = api.load('glove-twitter-25')

    # Step 3: Preprocess the data
    # Converts the text data into a Bag-of-Words representation and the labels into binary form
    # Returns the processed training, validation, and test data along with the fitted vectorizer and MultiLabelBinarizer
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb = preprocess_data(train_set, val_set, test_set, wv)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Step 4: Initialize the model, loss function, and optimizer
    # The model, loss function (BCELoss) or MultiLabelSoftMarginLoss, and optimizer (Adam) are created directly here
    input_size = X_train.shape[1]  # Number of input features
    output_size = y_train.shape[1]  # Number of output labels
    model = MLP(input_size, output_size, hidden_size, dropout_prob)  # Initialize the MLP model
    criterion = nn.MultiLabelSoftMarginLoss()  # Use MultiLabelSoftMarginLoss for multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Step 5: Train the model using the training and validation data and batch training
    trained_model = train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, batch_size, mlb)

    # Step 6: Evaluate the trained model on the test set (split from training data)
    test_loss = evaluate_model(trained_model, criterion, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Step 7: Evaluate multiple accuracy metrics on the test set
    test_accuracy = evaluate_accuracy(trained_model, X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 1-to-1 comparison of predicted vs. true labels for exact matches
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(X_test)
        predictions = (outputs >= 0.5).float()  # Convert probabilities to binary labels
        predicted_labels = mlb.inverse_transform(predictions.cpu().numpy())
        true_labels = mlb.inverse_transform(y_test.cpu().numpy())

    correct_predictions = sum(set(predicted_labels[i]) == set(true_labels[i]) for i in range(len(true_labels)))
    comparison_accuracy = correct_predictions / len(true_labels)
    print(f"1-to-1 Comparison Accuracy: {comparison_accuracy:.4f}")

    # Step 8: Preprocess the actual test file (from the test_file parameter) for submission
    test_df = pd.read_csv(test_file)  # Load the test file into a DataFrame
    X_test_submission_bow = vectorizer.transform(test_df['UTTERANCES'])  # Use the fitted vectorizer to transform the data
    # Convert text to GloVe embeddings for the test submission
    X_test_submission_glove = convert_text_to_glove_embeddings(test_df['UTTERANCES'], wv)

    # Concatenate BoW vectors with GloVe embeddings for the test submission
    X_test_submission_combined = np.hstack((X_test_submission_bow.toarray(), X_test_submission_glove))

    # Convert to PyTorch tensor
    X_test_submission_tensor = torch.tensor(X_test_submission_combined, dtype=torch.float32)

    # Step 9: Generate predictions for the test file and create a submission file
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