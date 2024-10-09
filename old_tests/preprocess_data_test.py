import unittest
import sys
import os
import pandas as pd

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.load_data import split_data
from data.preprocess import preprocess_data

class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        # Path to the hw1_train.csv file in the csv_files folder
        csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'csv_files', 'hw1_train.csv')
        
        # Load the actual training data from hw1_train.csv
        self.train_df = pd.read_csv(csv_file_path)

        # Ensure columns are as expected
        print(f"Columns in loaded data: {self.train_df.columns}")

        # Split the data for training, validation, and testing
        self.train_set, self.val_set, self.test_set = split_data(self.train_df)

    def test_preprocess_data(self):
        # Preprocess the data
        X_train, X_val, X_test, y_train, y_val, y_test, mlb = preprocess_data(self.train_set, self.val_set, self.test_set)

        # Check if shapes are correct
        self.assertEqual(X_train.shape[1], X_val.shape[1], "Feature dimension mismatch between train and validation.")
        self.assertEqual(X_train.shape[1], X_test.shape[1], "Feature dimension mismatch between train and test.")
        self.assertTrue(len(mlb.classes_) > 0, "The MultiLabelBinarizer didn't find any relations.")

if __name__ == "__main__":
    unittest.main()