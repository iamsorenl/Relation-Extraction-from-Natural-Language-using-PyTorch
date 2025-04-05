import sys
import spacy
import pandas as pd
import numpy as np
import torch
from utils.spacy import install_spacy_model, process_spacy_features  # Import from utils/spacy.py
from utils.kfold import perform_kfold_split  # Import from utils/kfold.py

# Overview of Implementation:
# 1. The input CSV files (training and test sets) are loaded into Pandas DataFrames.
# 2. Stratified K-fold cross-validation is used to create balanced training and validation splits.
# 3. The script uses spaCy for text processing and feature extraction.
# 4. Mean-pooling is applied to generate sentence-level embeddings from word vectors using the pre-trained spaCy model.
# 5. The Multi-Layer Perceptron (MLP) model is trained on these embeddings using K-fold cross-validation.
# 6. After training, the model is evaluated on the test set to generate predictions for core relations.
# 7. The script writes predictions to the provided submission CSV file containing predicted core relations for the test data.

def get_unique_labels(train_df):
    """Extracts unique labels from CORE RELATIONS column."""
    label_classes = sorted(set(train_df['CORE RELATIONS'].str.split().explode().unique()))
    return label_classes

def main(train_file, test_file, submission_file, num_folds=5, random_state=42, spacy_model_name='en_core_web_md'):

    # Install the spaCy model if it's not available
    install_spacy_model(spacy_model_name)

    # Load spaCy model
    nlp = spacy.load(spacy_model_name)
    
    # Load the data into a pandas DataFrame
    train_df = pd.read_csv(train_file)

    # Extract all unique labels from the training data
    label_classes = get_unique_labels(train_df)

    # Set input size and output size for the MLP model
    input_size = nlp.vocab.vectors_length  # Should resolve to 300 for static embeddings
    output_size = len(label_classes)  # Number of unique relations

    # Perform K-fold cross-validation on the training set and get the trained model
    trained_model = perform_kfold_split(train_df, nlp, num_folds=num_folds, random_state=random_state, input_size=input_size, output_size=output_size)

    # Load the test data into a pandas DataFrame
    test_df = pd.read_csv(test_file)

    # Process the test set using spaCy to get embeddings
    X_test = process_spacy_features(test_df['UTTERANCES'], nlp)

    # Run the trained model on the test set
    trained_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        test_outputs = trained_model(X_test)
        test_predictions = (test_outputs > 0.5).float()  # Binary classification

    # Convert the predictions to the correct label format and generate the submission
    generate_submission(test_predictions, test_df['ID'], label_classes, submission_file)


def generate_submission(predictions, ids, label_classes, submission_file):
    """
    Write predictions to the provided submission file.
    
    Parameters:
    - predictions: Tensor of model predictions (binary format)
    - ids: IDs from the test set to include in the submission
    - label_classes: List of unique class labels
    - submission_file: Path to the provided submission file where results should be saved
    """
    # Convert the predictions tensor to a NumPy array
    pred_array = predictions.cpu().numpy()

    submission_data = []
    submission_labels = []

    # For each prediction and ID, create the CORE RELATIONS string based on positive labels
    for pred, id_val in zip(pred_array, ids):
        # Get the labels corresponding to predicted 1's, and sort them alphabetically
        relations = [label_classes[i] for i in range(len(pred)) if pred[i] == 1]
        submission_data.append({'ID': id_val, 'CORE RELATIONS': ' '.join(sorted(relations))})
        submission_labels.append(relations)

    # Load the provided submission CSV
    submission_df = pd.read_csv(submission_file)

    # Update the 'CORE RELATIONS' column with the generated predictions
    submission_df['CORE RELATIONS'] = [row['CORE RELATIONS'] for row in submission_data]

    # Save the updated DataFrame back to the provided file
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file '{submission_file}' has been updated successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_data> <test_data> <submission_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    submission_file = sys.argv[3]
    main(train_file, test_file, submission_file)
