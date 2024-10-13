from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from utils.spacy import process_spacy_features
from models.mlp_model import MLP

def process_labels(labels):
    """Converts the CORE RELATIONS into multi-hot encoding for multi-label classification."""
    mlb = MultiLabelBinarizer()
    processed_labels = mlb.fit_transform(labels.str.split())
    return processed_labels, mlb.classes_


def perform_kfold_split(train_df, nlp, num_folds, random_state, input_size, output_size):
    """Perform Multilabel Stratified K-fold cross-validation."""

    # Convert CORE RELATIONS to multi-label format
    y, label_classes = process_labels(train_df['CORE RELATIONS'])

    # Initialize the Multilabel Stratified K-Fold object
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    fold_idx = 1
    trained_model = None
    max_length = None  # Placeholder for the maximum sequence length for padding

    # Perform the K-Fold split
    for train_index, val_index in mskf.split(train_df, y):
        print(f"\n--- Processing Fold {fold_idx}/{num_folds} ---")

        # Split the data into training and validation sets for this fold
        train_set = train_df.iloc[train_index]
        val_set = train_df.iloc[val_index]

        # Set max_length for padding based on the training set
        if max_length is None:
            max_length = max(len(seq) for seq in train_set['UTTERANCES'].apply(lambda x: nlp(x)))

        # Process spaCy features for training and validation sets
        X_train = process_spacy_features(train_set['UTTERANCES'], nlp, max_length=max_length)
        X_val = process_spacy_features(val_set['UTTERANCES'], nlp, max_length=max_length)

        # Print out the shapes and some sample data for validation
        print(f"Fold {fold_idx}:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"Sample X_train[0]: {X_train[0]}")
        print(f"Sample X_val[0]: {X_val[0]}")

        fold_idx += 1

    return trained_model