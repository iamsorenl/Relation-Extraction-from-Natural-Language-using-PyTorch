import unittest
import torch
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import the necessary functions from run.py
from run import initialize_model

class TestInitializeModel(unittest.TestCase):
    
    def test_initialize_model(self):
        input_size = 500  # Example feature size
        output_size = 10  # Example number of classes
        model, criterion, optimizer = initialize_model(input_size, output_size)
        
        # Test if model is initialized correctly
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(criterion, torch.nn.BCELoss)
        self.assertIsInstance(optimizer, torch.optim.Adam)

if __name__ == "__main__":
    unittest.main()