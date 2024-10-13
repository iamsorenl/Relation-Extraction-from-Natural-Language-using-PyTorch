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
        y_train, y_val = y[train_index], y[val_index]

        # Set max_length for padding based on the training set
        if max_length is None:
            max_length = max(len(seq) for seq in train_set['UTTERANCES'].apply(lambda x: nlp(x)))

        # Process spaCy features for training and validation sets
        X_train = process_spacy_features(train_set['UTTERANCES'], nlp, max_length=max_length)
        X_val = process_spacy_features(val_set['UTTERANCES'], nlp, max_length=max_length)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # Initialize the MLP model
        model = MLP(input_size=input_size, output_size=output_size)

        # Define loss function and optimizer
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 10
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
            outputs_val = model(X_val_tensor)
            y_pred = (outputs_val > 0.5).float()  # Convert logits to binary predictions

            # Calculate accuracy and F1-score
            accuracy = accuracy_score(y_val, y_pred.cpu().numpy())
            f1 = f1_score(y_val, y_pred.cpu().numpy(), average='micro')

            print(f"Fold {fold_idx} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        # After the last fold, return the trained model
        if fold_idx == num_folds:
            trained_model = model  # You could also store models from each fold if desired

        fold_idx += 1

    return trained_model