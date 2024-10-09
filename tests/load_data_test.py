import sys
import os
import pandas as pd

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.load_split_data import load_data

# Test the load_data function
def test_load_data():
    """
    Test the load_data function to ensure it correctly loads the data.
    """
    # Example paths to your CSV files (adjust these to point to your actual CSV files)
    train_file = "../csv_files/hw1_train.csv"
    test_file = "../csv_files/hw1_test.csv"

    # Call the load_data function to load the datasets
    train_df, test_df = load_data(train_file, test_file)
    
    # Assert that the returned objects are pandas DataFrames
    assert isinstance(train_df, pd.DataFrame), "train_df is not a DataFrame"
    assert isinstance(test_df, pd.DataFrame), "test_df is not a DataFrame"
    
    # Optional: Print the first few rows to visually inspect if needed
    print("Training Data:")
    print(train_df.head())
    
    print("\nTest Data:")
    print(test_df.head())

# Run the test if executed as a script
if __name__ == "__main__":
    test_load_data()