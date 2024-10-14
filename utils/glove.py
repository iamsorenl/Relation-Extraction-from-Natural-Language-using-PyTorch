import gensim.downloader as api
import numpy as np

# Function to load GloVe Twitter 300 embeddings using gensim
def load_glove_embeddings(model_name):
    """
    Load GloVe embeddings using gensim.downloader.

    Parameters:
    - model_name (str): The name of the pre-trained model to load.
    
    Returns:
    - embeddings_dict (gensim KeyedVectors): Gensim's KeyedVectors object with word embeddings.
    """
    print(api.info()['models'].keys())
    try:
        print(f"Loading GloVe embeddings: {model_name}")
        embeddings_dict = api.load(model_name)
        print(f"Successfully loaded {model_name} embeddings.")
    except ValueError as e:
        print(f"Error loading embeddings: {e}")
        embeddings_dict = None

    return embeddings_dict