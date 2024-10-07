import torch
import sys
import os

# Add the root directory to the Python path to access run.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import MLP class from run.py
from run import MLP

# Step 1: Define the input size, output size, and hidden size for testing
input_size = 5000  # Example input size (e.g., number of features in a Bag-of-Words or word embeddings)
output_size = 10   # Example output size (e.g., number of relations or classes)
hidden_size = 128  # Example hidden layer size

# Step 2: Instantiate the model
model = MLP(input_size=input_size, output_size=output_size, hidden_size=hidden_size)

# Step 3: Create a batch of fake input data (32 samples, each with input_size features)
batch_size = 32
fake_input = torch.randn(batch_size, input_size)  # Random data simulating input features

# Step 4: Pass the fake data through the model (forward pass)
output = model(fake_input)

# Step 5: Check the output shape
print(f"Output shape: {output.shape}")  # Should be (32, 10) if batch_size=32 and output_size=10

# Step 6: Check the first few outputs to ensure they are probabilities (values between 0 and 1)
print("Sample output:", output[0])  # First sample's output (should contain values between 0 and 1)