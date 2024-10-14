import spacy
import subprocess
import torch
import numpy as np
import sys

def install_spacy_model(spacy_model_name):
    """Install spaCy model if it's not already available."""
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"SpaCy model {spacy_model_name} not found. Installing...")
        subprocess.run([sys.executable, "-m", "spacy", "download", spacy_model_name])
        nlp = spacy.load(spacy_model_name)  # Load the model after installation
    return nlp