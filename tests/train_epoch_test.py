import unittest
import torch
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import the necessary functions from run.py
from run import train_one_epoch, initialize_model

class TestTrainOneEpoch(unittest.TestCase):
    def setUp(self):
        input_size = 500
        output_size = 10
        self.model, self.criterion, self.optimizer = initialize_model(input_size, output_size)
        self.device = torch.device('cpu')
    
    def test_train_one_epoch(self):
        X_train = torch.randn(32, 500)  # Batch of 32 samples, 500 features
        y_train = torch.randint(0, 2, (32, 10)).float()  # Binary labels
        
        # Train for one epoch
        loss = train_one_epoch(self.model, self.criterion, self.optimizer, X_train, y_train, self.device)
        self.assertIsInstance(loss, float)

if __name__ == "__main__":
    unittest.main()