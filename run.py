import sys
import torch
from models.mlp import MLP
from data.load_data import load_and_split_data

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

    # Load and split the training data into training, validation, and test sets
    train_set, val_set, test_set = load_and_split_data(train_file)

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size (split from training data): {len(test_set)}")

if __name__ == "__main__":
    # Example usage: python run.py hw1_train.csv hw1_test.csv
    if len(sys.argv) != 3:
        print("Usage: python run.py <train_data> <test_data>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)