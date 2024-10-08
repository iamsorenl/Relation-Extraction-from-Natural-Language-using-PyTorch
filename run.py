import sys
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# define hyperparameter
hidden_size = 128

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        """
        Initialize the MLP model with input size, output size, and an adjustable hidden layer size.

        Parameters:
        - input_size: int, the size of the input vector (e.g., the number of features in your data)
        - output_size: int, the number of output labels (e.g., the number of relations to predict)
        - hidden_size: int, the number of neurons in the hidden layer (default: 128)
        """
        super(MLP, self).__init__()

        # First fully connected layer (input to hidden layer)
        # This layer performs a linear transformation: Z1 = W1 * X + b1
        # Input: a vector of size (input_size,)
        # Output: a vector of size (hidden_size,)
        self.layer1 = nn.Linear(input_size, hidden_size)

        # Activation function: ReLU (Rectified Linear Unit)
        # This function is applied element-wise to the output of the first layer
        # It introduces non-linearity by outputting max(0, Z1), setting negative values to 0
        self.relu = nn.ReLU()

        # Second fully connected layer (hidden layer to output layer)
        # This layer performs another linear transformation: Z2 = W2 * A1 + b2
        # Input: a vector of size (hidden_size,)
        # Output: a vector of size (output_size,)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation function: Sigmoid
        # This is applied to the final output, converting the logits to probabilities
        # Sigmoid squashes the output values to the range [0, 1], making them interpretable as probabilities
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Defines the forward pass of the MLP model.
        
        Parameters:
        - x: torch.Tensor, the input tensor with shape (batch_size, input_size)
        
        Returns:
        - torch.Tensor, the output tensor with shape (batch_size, output_size) containing probabilities in [0, 1]
        """

        # Step 1: Apply the first linear layer (input to hidden layer)
        # x initially contains the input features, with shape (batch_size, input_size)
        # The layer1 performs the operation: Z1 = W1 * X + b1
        # After this step, x will have shape (batch_size, hidden_size)
        x = self.layer1(x)

        # Step 2: Apply the ReLU activation function
        # The ReLU function is applied element-wise to x, setting all negative values to 0
        # ReLU introduces non-linearity, which helps the network learn complex patterns
        # After this step, x still has shape (batch_size, hidden_size), but with non-negative values
        x = self.relu(x)

        # Step 3: Apply the second linear layer (hidden layer to output layer)
        # This layer performs another linear transformation: Z2 = W2 * A1 + b2
        # This step maps the hidden layer activations to the output space (logits for each label)
        # After this step, x will have shape (batch_size, output_size)
        x = self.output(x)

        # Step 4: Apply the Sigmoid activation function
        # The Sigmoid function is applied to each element of the output vector
        # Sigmoid squashes the output values to the range [0, 1], converting the raw logits to probabilities
        # Each value in the final output represents the probability of a particular label being active
        # After this step, x still has shape (batch_size, output_size), but all values are in [0, 1]
        x = self.sigmoid(x)

        # Return the output tensor, containing probabilities for each label
        return x
    
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

def preprocess_data(train_set, val_set, test_set):
    """
    Preprocesses the text and labels for the model.
    
    Parameters:
    - train_set: pd.DataFrame, the training data
    - val_set: pd.DataFrame, the validation data
    - test_set: pd.DataFrame, the test data
    
    Returns:
    - X_train: Bag-of-Words representation of the training set
    - X_val: Bag-of-Words representation of the validation set
    - X_test: Bag-of-Words representation of the test set
    - y_train: Binary label matrix for the training set
    - y_val: Binary label matrix for the validation set
    - y_test: Binary label matrix for the test set
    - mlb: MultiLabelBinarizer instance fitted on the training data
    """
    
    # Step 1: Preprocess the text using Bag-of-Words
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_set['UTTERANCES'])
    X_val = vectorizer.transform(val_set['UTTERANCES'])
    X_test = vectorizer.transform(test_set['UTTERANCES'])
    
    # Step 2: Preprocess the labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Fit the binarizer only on the training labels
    train_labels = train_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    val_labels = val_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    test_labels = test_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    
    y_train = mlb.fit_transform(train_labels)  # Fit the binarizer on training labels only
    y_val = mlb.transform(val_labels)  # Transform validation labels
    y_test = mlb.transform(test_labels)  # Transform test labels
    
    return X_train, X_val, X_test, y_train, y_val, y_test, mlb