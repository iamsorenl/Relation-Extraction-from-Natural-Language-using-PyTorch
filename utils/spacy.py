import spacy
import subprocess
import sys
import numpy as np

def install_spacy_model(spacy_model_name):
    """Function to install spaCy model if it's not already installed."""
    try:
        # Attempt to load the model to check if it's already installed
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"SpaCy model {spacy_model_name} not found. Installing...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
        nlp = spacy.load(spacy_model_name)  # Load the model after installation
    return nlp

def process_spacy_features(text_data, nlp):
    """Helper function to convert text to spaCy embeddings and extract additional features."""
    embeddings = []
    pos_tags = []
    named_entities = []
    
    # Process each document using spaCy's pipeline
    for doc in nlp.pipe(text_data):
        embeddings.append(doc.vector)  # Get the mean-pooled vector for the document

        # Extract Part-of-Speech (POS) tags
        pos_tags.append([token.pos_ for token in doc])

        # Extract Named Entities (NER)
        named_entities.append([(ent.text, ent.label_) for ent in doc.ents])

    return np.array(embeddings), pos_tags, named_entities