import spacy
import subprocess
import sys
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence  # Use PyTorch for padding

def install_spacy_model(spacy_model_name):
    """Install spaCy model if it's not already available."""
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"SpaCy model {spacy_model_name} not found. Installing...")
        subprocess.run([sys.executable, "-m", "spacy", "download", spacy_model_name])
        nlp = spacy.load(spacy_model_name)  # Load the model after installation
    return nlp

def process_spacy_features(text_data, nlp, max_length=None):
    """Extract embeddings, POS tags, and dependency relations using spaCy."""
    embeddings = []
    pos_tags = []
    dep_relations = []

    # Process each document using spaCy's pipeline
    for doc in nlp.pipe(text_data):
        embeddings.append(doc.vector)  # Mean-pooled vector for embeddings
        pos_tags.append([token.pos_ for token in doc])  # POS tags
        dep_relations.append([token.dep_ for token in doc])  # Dependency relations

    # Set max_length to the longest sequence if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in pos_tags)

    # Convert POS tags and dependency relations to numeric values for one-hot encoding
    pos_encoded, dep_encoded = one_hot_encode_features(pos_tags, dep_relations, nlp)

    # Pad POS tags and dependency relations
    pos_tags_padded = pad_sequence([torch.tensor(seq) for seq in pos_encoded], batch_first=True, padding_value=0)
    dep_relations_padded = pad_sequence([torch.tensor(seq) for seq in dep_encoded], batch_first=True, padding_value=0)

    # Repeat the embedding vector for each token (to match the sequence length)
    embeddings = torch.tensor(embeddings).unsqueeze(1).repeat(1, max_length, 1)  # Shape: (batch_size, max_length, embedding_dim)

    # Concatenate embeddings with encoded and padded POS tags and dependencies
    combined_features = torch.cat([embeddings, pos_tags_padded, dep_relations_padded], dim=2)

    return combined_features


def one_hot_encode_features(pos_tags, dep_relations, nlp):
    """One-hot encode POS tags and dependency relations sentence by sentence."""
    # Fetch available POS and dependency labels from spaCy
    pos_labels = nlp.pipe_labels['tagger']  # Part-of-speech tag labels
    dep_labels = nlp.pipe_labels['parser']  # Dependency relation labels

    # Create OneHotEncoder for POS tags and dependency relations
    pos_encoder = OneHotEncoder(categories=[pos_labels], sparse_output=False, handle_unknown='ignore')
    dep_encoder = OneHotEncoder(categories=[dep_labels], sparse_output=False, handle_unknown='ignore')

    pos_encoded_sentences = []
    dep_encoded_sentences = []

    # Encode POS tags and dependencies for each sentence separately
    for pos_seq, dep_seq in zip(pos_tags, dep_relations):
        pos_encoded = pos_encoder.fit_transform(np.array(pos_seq).reshape(-1, 1))  # One-hot encode POS tags for each sentence
        dep_encoded = dep_encoder.fit_transform(np.array(dep_seq).reshape(-1, 1))  # One-hot encode dependency relations for each sentence

        # Append encoded sentences back to list
        pos_encoded_sentences.append(pos_encoded)
        dep_encoded_sentences.append(dep_encoded)

    return pos_encoded_sentences, dep_encoded_sentences  # Return lists of encoded sequences for each sentence