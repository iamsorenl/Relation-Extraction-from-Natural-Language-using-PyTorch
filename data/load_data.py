import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_file, test_file):
    """
    Load the training and test data from CSV files.
    
    Parameters:
    - train_file: str, path to the training data CSV file
    - test_file: str, path to the test data CSV file
    
    Returns:
    - train_df: pd.DataFrame, the loaded training data
    - test_df: pd.DataFrame, the loaded test data
    """
    # Load the CSV files into pandas DataFrames
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    return train_df, test_df

def split_data(train_df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    - train_df: pd.DataFrame, the loaded dataset
    - test_size: float, the proportion of the data to be used as the test set
    - val_size: float, the proportion of the remaining train set to be used as validation
    - random_state: int, seed for reproducibility
    
    Returns:
    - train_set: pd.DataFrame, training data
    - val_set: pd.DataFrame, validation data
    - test_set: pd.DataFrame, test data
    """
    # First, split into train+validation and test
    train_val_set, test_set = train_test_split(train_df, test_size=test_size, random_state=random_state)

    # Then, split the remaining train+validation set into training and validation
    train_set, val_set = train_test_split(train_val_set, test_size=val_size, random_state=random_state)
    
    return train_set, val_set, test_set