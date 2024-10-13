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

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # Initialize the MLP model for this fold
        model = MLP(input_size=input_size, output_size=output_size)

        # Define loss function and optimizer
        criterion = nn.MultiLabelSoftMarginLoss()  # For multi-label classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the MLP model
        num_epochs = 100 
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f'Fold {fold_idx}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model on validation set
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_val_tensor)
            y_pred = (y_pred_tensor > 0.5).float()  # Convert logits to binary predictions

            # Calculate accuracy and F1-score
            accuracy = accuracy_score(y_val, y_pred.cpu().numpy())
            f1 = f1_score(y_val, y_pred.cpu().numpy(), average='micro')

            print(f"Fold {fold_idx} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

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