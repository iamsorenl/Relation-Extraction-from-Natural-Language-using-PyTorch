from sklearn.decomposition import PCA
import numpy as np
import torch
from utils.ngram import extract_character_ngrams

def process_combined_features(text_data, nlp, glove_embeddings, output_dim):
    """
    Extract combined SpaCy, GloVe, and N-gram embeddings for text data and reduce to desired dimensions using PCA.
    
    Parameters:
    - text_data: List of text documents.
    - nlp: Loaded spaCy model for embeddings.
    - glove_embeddings: GloVe embeddings in gensim format.
    - output_dim: Target dimensionality, typically passed from MLP's hidden size.

    Returns:
    - PyTorch tensor of reduced embeddings (dim = output_dim).
    """
    combined_embeddings = []
    glove_dim = glove_embeddings.vector_size

    # Process N-grams
    ngram_features = extract_character_ngrams(text_data, ngram_range=(2, 4))  # Adjust n-gram range as needed
    print(f"N-gram features shape: {ngram_features.shape}")

    # Process each document using spaCy and combine with GloVe
    for i, doc in enumerate(nlp.pipe(text_data)):
        spacy_vector = doc.vector
        words = [token.text for token in doc]
        glove_vectors = np.mean([glove_embeddings[word] if word in glove_embeddings else np.zeros(glove_dim) for word in words], axis=0)

        # Get the corresponding N-gram features for the current document
        ngram_vector = ngram_features[i].toarray().flatten()

        # Combine SpaCy, GloVe, and N-gram features
        combined_vector = np.concatenate((spacy_vector, glove_vectors, ngram_vector))
        combined_embeddings.append(combined_vector)

        print(f"Document {i}: Combined vector shape: {combined_vector.shape}")

    # Apply PCA to reduce the dimensionality of the combined embeddings
    combined_embeddings = np.array(combined_embeddings)
    print(f"Original combined embeddings shape: {combined_embeddings.shape}")

    # Apply PCA to reduce to the desired output dimension
    pca = PCA(n_components=output_dim)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    return torch.tensor(reduced_embeddings, dtype=torch.float32)