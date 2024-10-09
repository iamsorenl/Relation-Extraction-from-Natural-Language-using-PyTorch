import unittest
import torch
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.train import evaluate_model
from initialize.bce_adam import initialize_mlp_bce_adam

class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        input_size = 500
        output_size = 10
        self.model, self.criterion, _ = initialize_mlp_bce_adam(input_size, output_size)
        self.device = torch.device('cpu')
    
    def test_evaluate_model(self):
        # Create a small batch of validation data
        X_val = torch.randn(32, 500)  # Batch of 32 samples, 500 features
        y_val = torch.randint(0, 2, (32, 10)).float()  # Binary labels
        
        # Evaluate the model
        val_loss = evaluate_model(self.model, self.criterion, X_val, y_val, self.device)
        
        # Test if the validation loss is a float
        self.assertIsInstance(val_loss, float)

if __name__ == "__main__":
    unittest.main()