import sys
import spacy
import pandas as pd
import numpy as np
from utils.spacy import install_spacy_model, process_spacy_features  # Import from utils/spacy.py
from utils.kfold import perform_kfold_split  # Import from utils/kfold.py

# Overview of Implementation:
# 1. The input CSV files are loaded into Pandas DataFrames.
# 2. Stratified K-fold cross-validation is used to create balanced training and validation splits.
# 3. The script uses spaCy for text processing and feature extraction.
# 4. Mean-pooling is applied to generate sentence-level embeddings using the word vectors from spaCy.
# 5. Additional features such as POS tags and Named Entities are extracted.
# 6. Evaluation metrics such as Accuracy and F1-Score are calculated for each fold.

def main(train_file, test_file, num_folds=5, random_state=42, spacy_model_name='en_core_web_md'):

    # Install the spaCy model if it's not available
    install_spacy_model(spacy_model_name)

    # Load spaCy model
    nlp = spacy.load(spacy_model_name)

    # Debug: Check the pipeline components
    print(f"Pipeline components: {nlp.pipe_names}")
    
    # Load the data into a pandas DataFrame
    train_df = pd.read_csv(train_file)

    # Set input size and output size for the MLP model
    input_size = nlp.vocab.vectors_length # Should resolve to 300 for static embeddings
    output_size = len(train_df['CORE RELATIONS'].str.split().explode().unique())  # Number of unique relations

    # Perform K-fold cross-validation on the training set and get the trained model
    trained_model = perform_kfold_split(train_df, nlp, num_folds=num_folds, random_state=random_state, input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    # Example usage: python run.py <train_data> <test_data>
    if len(sys.argv) != 3:
        print("Usage: python run.py <train_data> <test_data>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)