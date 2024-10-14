import spacy
import subprocess
import torch
import numpy as np

def install_spacy_model(spacy_model_name):
    """Install spaCy model if it's not already available."""
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"SpaCy model {spacy_model_name} not found. Installing...")
        subprocess.run([sys.executable, "-m", "spacy", "download", spacy_model_name])
        nlp = spacy.load(spacy_model_name)  # Load the model after installation
    return nlp

def process_spacy_features(text_data, nlp):
    """Extract word embeddings using spaCy."""
    embeddings = []

    # Process each document using spaCy's pipeline
    for doc in nlp.pipe(text_data):
        embeddings.append(doc.vector)  # Use spaCy's mean-pooled vector for each document

    # Convert list of numpy arrays to a single numpy array before creating PyTorch tensor
    embeddings = torch.tensor(np.array(embeddings))  # Convert list of numpy arrays to a single tensor

    return embeddings