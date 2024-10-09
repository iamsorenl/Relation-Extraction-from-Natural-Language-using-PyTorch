from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_data(train_set, val_set, test_set):
    """
    Preprocesses the text and labels for the model.
    
    Parameters:
    - train_set: pd.DataFrame, the training data
    - val_set: pd.DataFrame, the validation data
    - test_set: pd.DataFrame, the test data
    
    Returns:
    - X_train: Bag-of-Words representation of the training set
    - X_val: Bag-of-Words representation of the validation set
    - X_test: Bag-of-Words representation of the test set
    - y_train: Binary label matrix for the training set
    - y_val: Binary label matrix for the validation set
    - y_test: Binary label matrix for the test set
    - mlb: MultiLabelBinarizer instance fitted on the training data
    """
    
    # Step 1: Preprocess the text using Bag-of-Words
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_set['UTTERANCES'])
    X_val = vectorizer.transform(val_set['UTTERANCES'])
    X_test = vectorizer.transform(test_set['UTTERANCES'])
    
    # Step 2: Preprocess the labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Fit the binarizer only on the training labels
    train_labels = train_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    val_labels = val_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    test_labels = test_set['CORE RELATIONS'].apply(lambda x: sorted(x.split()))
    
    y_train = mlb.fit_transform(train_labels)  # Fit the binarizer on training labels only
    y_val = mlb.transform(val_labels)  # Transform validation labels
    y_test = mlb.transform(test_labels)  # Transform test labels
    
    return X_train, X_val, X_test, y_train, y_val, y_test, mlb