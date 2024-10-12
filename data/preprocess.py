from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from nltk.tokenize import word_tokenize

# Define the convert_text_to_glove_embeddings function globally
def convert_text_to_glove_embeddings(text_data, wv):
    text_embeddings = []
    for text in text_data:
        token_embeddings = []
        for token in word_tokenize(text.lower()):
            if token in wv:
                token_embeddings.append(wv[token])
        if len(token_embeddings) > 0:
            text_embedding = np.mean(np.array(token_embeddings), axis=0)
        else:
            text_embedding = np.zeros(wv.vector_size)  # Zero vector if no tokens found
        text_embeddings.append(text_embedding)
    return np.array(text_embeddings)

def preprocess_data(train_set, val_set, test_set, wv):
    """
    Preprocesses the text and labels for the model.
    
    Parameters:
    - train_set: pd.DataFrame, the training data
    - val_set: pd.DataFrame, the validation data
    - test_set: pd.DataFrame, the test data
    - wv: Preloaded GloVe model (gensim KeyedVectors)
    
    Returns:
    - X_train: Concatenated Bag-of-Words + GloVe representation of the training set
    - X_val: Concatenated Bag-of-Words + GloVe representation of the validation set
    - X_test: Concatenated Bag-of-Words + GloVe representation of the test set
    - y_train: Binary label matrix for the training set
    - y_val: Binary label matrix for the validation set
    - y_test: Binary label matrix for the test set
    - mlb: MultiLabelBinarizer instance fitted on the training data
    - vectorizer: The fitted CountVectorizer for BoW
    """
    
    # Step 1: Preprocess the text using Bag-of-Words
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(train_set['UTTERANCES'])
    X_val_bow = vectorizer.transform(val_set['UTTERANCES'])
    X_test_bow = vectorizer.transform(test_set['UTTERANCES'])
    
    # Step 2: Convert text to GloVe embeddings
    X_train_glove = convert_text_to_glove_embeddings(train_set['UTTERANCES'], wv)
    X_val_glove = convert_text_to_glove_embeddings(val_set['UTTERANCES'], wv)
    X_test_glove = convert_text_to_glove_embeddings(test_set['UTTERANCES'], wv)

    # Step 3: Concatenate the BoW vectors with GloVe embeddings
    X_train = np.hstack((X_train_bow.toarray(), X_train_glove))
    X_val = np.hstack((X_val_bow.toarray(), X_val_glove))
    X_test = np.hstack((X_test_bow.toarray(), X_test_glove))

    # Step 4: Preprocess the labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    train_labels = train_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    val_labels = val_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    test_labels = test_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))

    y_train = mlb.fit_transform(train_labels)  # Fit the binarizer on training labels only
    y_val = mlb.transform(val_labels)  # Transform validation labels
    y_test = mlb.transform(test_labels)  # Transform test labels

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb