import sys
import os
import pandas as pd

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the load_data and split_data functions from run.py
from data.load_split_data import load_data, split_data

# Test the split_data function
def test_split_data():
    """
    Test the split_data function to ensure it correctly splits the data into training, validation, and test sets.
    """
    # Example paths to your CSV files (adjust these to point to your actual CSV files)
    train_file = "../csv_files/hw1_train.csv"
    test_file = "../csv_files/hw1_test.csv"

    # Load the data (we only need the training data for splitting)
    train_df, _ = load_data(train_file, test_file)
    
    # Split the data into training, validation, and test sets
    train_set, val_set, test_set = split_data(train_df, test_size=0.2, val_size=0.25)
    
    # Assertions to check if the split is correct
    total_size = len(train_set) + len(val_set) + len(test_set)
    assert total_size == len(train_df), f"Total number of samples after splitting is incorrect: expected {len(train_df)}, got {total_size}"
    
    # Check if the proportions are approximately correct
    test_size_expected = int(0.2 * len(train_df))
    remaining_data = len(train_df) - test_size_expected
    val_size_expected = int(0.25 * remaining_data)
    
    assert abs(len(test_set) - test_size_expected) <= 1, f"Test set size is incorrect: expected {test_size_expected}, got {len(test_set)}"
    assert abs(len(val_set) - val_size_expected) <= 1, f"Validation set size is incorrect: expected {val_size_expected}, got {len(val_set)}"
    
    # Optional: Print the sizes and inspect the data
    print(f"Training Set Size: {len(train_set)}")
    print(f"Validation Set Size: {len(val_set)}")
    print(f"Test Set Size: {len(test_set)}")
    
    print("Training Set Sample:")
    print(train_set.head())
    
    print("\nValidation Set Sample:")
    print(val_set.head())
    
    print("\nTest Set Sample:")
    print(test_set.head())

# Run the test if executed as a script
if __name__ == "__main__":
    test_split_data()