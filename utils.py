# utils.py
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')


# Function for text preprocessing
def process_text(text):
    """
    Processes the input text by performing tokenization, normalization, stopword removal, and lemmatization.

    Input: 
       text - the text to be processed

    Returns: 
       str: The processed text with tokens joined into a single string.
    """
    # Tokenize the input text into individual words
    token = word_tokenize(text)
    # Normalize text by converting to lowercase and filtering out non-alphabetic tokens
    token = [word.lower() for word in token if word.isalpha()]
    # Remove common stopwords to reduce noise in the data
    stop_words = set(stopwords.words('english'))
    token = [word for word in token if word not in stop_words]
    # Apply lemmatization to reduce words to their base or root form
    lemma = WordNetLemmatizer()
    token = [lemma.lemmatize(word) for word in token]
    # Join tokens back into a single string with space separation
    return ' '.join(token)


"""

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the 
importance of a word in a document relative to a collection of documents. 
It combines the term frequency (how often a word appears) with the inverse document frequency (how common or rare a word is across documents).


Word2Vec is a word embedding technique that represents words in a continuous vector space based on their context 
within a corpus. It captures semantic relationships by transforming words into vectors that reflect 
their meanings and usage patterns.


Bag of Words (BoW) represents text data by converting documents into vectors based on word frequencies, 
ignoring grammar and word order. Each document is represented 
as a vector of word counts or binary values for the presence of words in a fixed vocabulary.

"""


def get_tfidf_embeddings(patterns):
    """
    Converts a list of text patterns to TF-IDF vectors.

    Parameters:
    - patterns: list of str, the text patterns to convert.

    Returns:
    - X: np.ndarray, the TF-IDF vectors for the patterns.
    - vectorizer: TfidfVectorizer, the fitted TF-IDF vectorizer.
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the patterns and transform the patterns to TF-IDF vectors
    X = vectorizer.fit_transform(patterns).toarray()

    return X, vectorizer


def get_word2vec_embeddings(patterns):
    """
    Converts a list of text patterns to Word2Vec vectors by averaging the word vectors.

    Parameters:
    - patterns: list of str, the text patterns to convert.

    Returns:
    - np.ndarray, the averaged Word2Vec vectors for the patterns.
    - model: Word2Vec, the trained Word2Vec model.
    """
    # Tokenize patterns into words
    tokenized_patterns = [process_text(pattern).split()
                          for pattern in patterns]

    # Train the Word2Vec model on the tokenized patterns
    model = Word2Vec(sentences=tokenized_patterns,
                     vector_size=100, window=5, min_count=1, sg=0)

    # Access the word vectors from the trained model
    word_vectors = model.wv

    # Compute the average vector for each pattern
    vectors = []
    for pattern in tokenized_patterns:
        vec = np.mean([word_vectors[word]
                      for word in pattern if word in word_vectors], axis=0)
        vectors.append(vec)

    return np.array(vectors), model


def get_bow_embeddings(patterns):
    """
    Converts a list of text patterns to Bag of Words vectors.

    Parameters:
    - patterns: list of str, the text patterns to convert.

    Returns:
    - np.ndarray, the BoW vectors for the patterns.
    - vectorizer: CountVectorizer, the fitted CountVectorizer model.
    """
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the patterns to BoW vectors
    X_bow = vectorizer.fit_transform(patterns).toarray()

    return X_bow, vectorizer

def save_embedding_models(vectorizer=None, label_encoder=None, tag_responses=None, word2vec_model=None, bow_model=None):
    """
    Save the vectorizer, label encoder, tag responses, and embedding models to files.

    Parameters:
    - vectorizer: TfidfVectorizer, optional, the TF-IDF vectorizer to save.
    - label_encoder: LabelEncoder, optional, the label encoder to save.
    - tag_responses: dict, optional, the tag responses dictionary to save.
    - word2vec_model: Word2Vec, optional, the Word2Vec model to save.
    - bow_model: bow, optional, the bow model to save.
    """
    if vectorizer:
        joblib.dump(vectorizer, 'vectorizer.pkl')
    if label_encoder:
        joblib.dump(label_encoder, 'label_encoder.pkl')
    if tag_responses:
        joblib.dump(tag_responses, 'tag_responses.pkl')
    if word2vec_model:
        word2vec_model.save('word2vec_model.bin')
    if bow_model:
        joblib.dump(bow_model, 'bow_model.pkl')

