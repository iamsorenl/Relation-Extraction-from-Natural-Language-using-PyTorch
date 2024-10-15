from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.spacy import process_spacy_features
from models.mlp_model import MLP

def process_labels(labels):
    """Converts CORE RELATIONS into multi-hot encoding for multi-label classification."""
    mlb = MultiLabelBinarizer()
    processed_labels = mlb.fit_transform(labels.str.split())
    return processed_labels

def perform_kfold_split(train_df, nlp, num_folds, random_state, input_size, output_size):
    """Perform Multilabel Stratified K-fold cross-validation."""
    
    # Convert CORE RELATIONS to multi-label format
    y = process_labels(train_df['CORE RELATIONS'])

    # Initialize the Multilabel Stratified K-Fold object
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    fold_idx = 1
    trained_model = None

    # Initialize variables for best weights and lowest loss tracking
    lowest_loss = float('inf')
    best_weights = None

    # Lists to store accuracy and F1-score for each fold
    accuracies = []
    f1_scores = []
    val_losses = []

    # Perform the K-Fold split
    for train_index, val_index in mskf.split(train_df, y):
        print(f"\n--- Processing Fold {fold_idx}/{num_folds} ---")

        # Split the data into training and validation sets for this fold
        train_set = train_df.iloc[train_index]
        val_set = train_df.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Process spaCy embeddings for training and validation sets
        X_train = process_spacy_features(train_set['UTTERANCES'], nlp)
        X_val = process_spacy_features(val_set['UTTERANCES'], nlp)

        # Convert labels to PyTorch tensors
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # Initialize the MLP model
        model = MLP(input_size=input_size, output_size=output_size)

        # Define loss function and optimizer
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model
        num_epochs = 300
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f'Fold {fold_idx}, Epoch [{epoch+1}/{num_epochs}], MLSM Loss: {loss.item():.4f}')

            # Save the best model weights if current loss is lower than previous best
            if loss.item() < lowest_loss:
                lowest_loss = loss.item()
                best_weights = model.state_dict()  # Store the best model weights

        # Evaluate the model on validation set
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            y_pred = (outputs_val > 0.5).float()  # Convert logits to binary predictions

            # Calculate accuracy and F1-score and validation
            accuracy = accuracy_score(y_val, y_pred.cpu().numpy())
            f1 = f1_score(y_val, y_pred.cpu().numpy(), average='micro')
            val_loss = criterion(outputs_val, y_val_tensor)

            print(f"Fold {fold_idx} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Validation Loss: {val_loss.item():.4f}")

            val_losses.append(val_loss)
            accuracies.append(accuracy)
            f1_scores.append(f1)
        
        # After each fold, restore the best weights
        if best_weights is not None:
            model.load_state_dict(best_weights)

        # After the last fold, return the trained model
        trained_model = model

        fold_idx += 1

    # Print average scores across all folds
    mean_val_losses = np.mean(val_losses)
    std_val_losses = np.std(val_losses)
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_accuracy = np.std(accuracies)
    std_f1 = np.std(f1_scores)

    print(f"\nAverage Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Average F1-Score: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"Average Val Loss: {mean_val_losses:.4f} (±{std_val_losses:.4f})")

    return trained_model