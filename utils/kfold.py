from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from utils.spacy import process_spacy_features

def process_labels(labels):
    """Converts the CORE RELATIONS into multi-hot encoding for multi-label classification."""
    mlb = MultiLabelBinarizer()
    processed_labels = mlb.fit_transform(labels.str.split())  # Assuming CORE RELATIONS are space-separated
    return processed_labels, mlb.classes_


def perform_kfold_split(train_df, nlp, num_folds, random_state):
    """Perform Multilabel Stratified K-fold cross-validation."""

    # Convert CORE RELATIONS to multi-label format
    y, label_classes = process_labels(train_df['CORE RELATIONS'])

    # Initialize the Multilabel Stratified K-Fold object
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_idx = 1
    
    # Perform the K-Fold split
    for train_index, val_index in mskf.split(train_df, y):
        print(f"Processing Fold {fold_idx}/{num_folds}")

        # Split the data into training and validation sets for this fold
        train_set, val_set = train_df.iloc[train_index], train_df.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert text to embeddings and extract additional features
        X_train, train_pos_tags, train_named_entities = process_spacy_features(train_set['UTTERANCES'], nlp)
        X_val, val_pos_tags, val_named_entities = process_spacy_features(val_set['UTTERANCES'], nlp)

        print(f"Processed training data shape for Fold {fold_idx}: {X_train.shape}")
        print(f"Processed validation data shape for Fold {fold_idx}: {X_val.shape}")

        # Store the processed data for this fold
        fold_results.append({
            'fold_idx': fold_idx,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'train_pos_tags': train_pos_tags,
            'val_pos_tags': val_pos_tags,
            'train_named_entities': train_named_entities,
            'val_named_entities': val_named_entities,
        })

        fold_idx += 1

    return fold_results, label_classes  # Return processed data and label classes for each fold