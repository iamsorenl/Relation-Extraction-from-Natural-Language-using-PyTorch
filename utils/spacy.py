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

    # Get unique POS and dependency tags from spaCy
    pos_tag_map = {pos: i for i, pos in enumerate(nlp.pipe_labels['tagger'])}
    dep_rel_map = {dep: i for i, dep in enumerate(nlp.pipe_labels['parser'])}
    
    # Process each document using spaCy's pipeline
    for doc in nlp.pipe(text_data):
        embeddings.append(torch.tensor(doc.vector))  # Convert embeddings to torch tensor

        # Convert POS tags and dependency relations to integer mappings
        pos_tags.append([pos_tag_map[token.pos_] for token in doc])
        dep_relations.append([dep_rel_map[token.dep_] for token in doc])

    # Set max_length to the longest sequence if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in pos_tags)

    # Pad POS tags and dependency relations
    pos_tags_padded = pad_sequence([torch.tensor(seq) for seq in pos_tags], batch_first=True, padding_value=0)
    dep_relations_padded = pad_sequence([torch.tensor(seq) for seq in dep_relations], batch_first=True, padding_value=0)

    # One-hot encode POS tags and dependency relations
    pos_encoded, dep_encoded = one_hot_encode_features(pos_tags_padded, dep_relations_padded)

    # Ensure embeddings are repeated across the same max_length as POS tags and dependencies
    embeddings_padded = [emb.unsqueeze(0).repeat(max_length, 1) for emb in embeddings]

    # Concatenate embeddings, one-hot encoded POS tags, and dependencies
    combined_features = [torch.cat([emb, pos_enc, dep_enc], dim=1) 
                         for emb, pos_enc, dep_enc in zip(embeddings_padded, pos_encoded, dep_encoded)]

    return torch.stack(combined_features)

def one_hot_encode_features(pos_tags, dep_relations):
    """One-hot encode POS tags and dependency relations."""
    pos_tags_flat = pos_tags.view(-1).cpu().numpy()  # Flatten the tensor
    dep_relations_flat = dep_relations.view(-1).cpu().numpy()  # Flatten the tensor

    # One-hot encode the POS tags and dependency relations
    pos_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    dep_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    pos_encoded = pos_encoder.fit_transform(np.array(pos_tags_flat).reshape(-1, 1))
    dep_encoded = dep_encoder.fit_transform(np.array(dep_relations_flat).reshape(-1, 1))

    # Reshape the one-hot encoded arrays back to the original structure
    pos_encoded_tensor = torch.tensor(pos_encoded).view(pos_tags.size(0), pos_tags.size(1), -1)
    dep_encoded_tensor = torch.tensor(dep_encoded).view(dep_relations.size(0), dep_relations.size(1), -1)

    return pos_encoded_tensor, dep_encoded_tensor