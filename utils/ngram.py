from sklearn.feature_extraction.text import TfidfVectorizer

# Function to extract character n-grams
def extract_character_ngrams(text_data, ngram_range=(3, 6)):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    return vectorizer.fit_transform(text_data)