import torch
import pandas as pd

def create_submission_file(model, X_test, mlb, output_file='submission.csv', threshold=0.5):
    """
    Create a submission file for Kaggle.

    Parameters:
    - model: The trained MLP model.
    - X_test: Test data (features as a PyTorch tensor).
    - mlb: MultiLabelBinarizer used for transforming labels.
    - output_file: str, the name of the submission file to create.
    - threshold: float, the threshold for converting predicted probabilities to binary labels.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Move the test data to the CPU for processing (if on GPU)
    X_test = X_test.cpu()
    
    with torch.no_grad():  # Disable gradient computation
        # Perform the forward pass to get predictions
        outputs = model(X_test)
        # Convert predicted probabilities to binary labels using the threshold
        predictions = (outputs >= threshold).float()
        # Transform the predictions back to the original label format
        predicted_labels = mlb.inverse_transform(predictions.cpu().numpy())
    
    # Create a DataFrame for submission
    submission = pd.DataFrame({
        'ID': range(1, len(predicted_labels) + 1),  # Create an ID column
        'Core Relations': [' '.join(labels) if labels else '' for labels in predicted_labels]  # Join labels
    })
    
    # Save the submission DataFrame to a CSV file
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")