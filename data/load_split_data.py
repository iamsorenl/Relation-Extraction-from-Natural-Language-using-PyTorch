import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_file, test_size=0.2, val_size=0.2, random_state=42):
    """
    Load the dataset from a CSV file and split it into training, validation, and test sets.
    
    Parameters:
    - data_file: str, path to the data CSV file
    - test_size: float, proportion of the data to be used as the test set
    - val_size: float, proportion of the remaining data to be used as validation
    - random_state: int, seed for reproducibility
    
    Returns:
    - train_set: pd.DataFrame, training data
    - val_set: pd.DataFrame, validation data
    - test_set: pd.DataFrame, test data
    """
    # Load the data into a pandas DataFrame
    data_df = pd.read_csv(data_file)
    
    # First, split into train+validation and test
    train_val_set, test_set = train_test_split(data_df, test_size=test_size, random_state=random_state)

    # Then, split the remaining train+validation set into training and validation
    train_set, val_set = train_test_split(train_val_set, test_size=val_size, random_state=random_state)
    
    return train_set, val_set, test_set