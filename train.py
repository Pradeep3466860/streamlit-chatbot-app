# Import necessary packages
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore
import joblib
from tensorflow.keras import layers, models  # type: ignore
from models import bot_ann_model, bot_svm_model, bot_lstm_model
from utils import process_text, get_tfidf_embeddings, get_word2vec_embeddings, get_bow_embeddings, save_embedding_models

# Load intent data from JSON file
with open('Data/intents.json') as file:
    intents = json.load(file)

# Initialise lists to store patterns, responses, and tags
patterns = []
responses = []
tags = []

# Dictionary to map tags to their respective responses
tag_responses = {}

# Iterate over each intent in the JSON data
for intent in intents['intents']:
    tag = intent['tag']
    tag_responses[tag] = intent['responses']  # Store responses for the tag
    # Process each pattern in the current intent
    for pattern in intent['patterns']:
        # Preprocess the text pattern
        patterns.append(process_text(pattern))
        # Use the first response for now
        responses.append(intent['responses'][0])
        tags.append(tag)  # Append the tag associated with the pattern

# Encode tags into integers for model training
label_encoder = LabelEncoder()
tags_encoded = label_encoder.fit_transform(
    tags)  # Transform tags to numerical labels
num_classes = len(label_encoder.classes_)  # Number of unique classes (tags)

# Get embeddings using different methods
X_tfidf, vectorizer = get_tfidf_embeddings(patterns)
X_word2vec, word2vec_model = get_word2vec_embeddings(patterns)
X_bow, bow_model = get_bow_embeddings(patterns)

# Convert tag labels to categorical format for training
y = to_categorical(tags_encoded, num_classes=num_classes)

# Split data for training and testing with TF-IDF embeddings
tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# Train ANN model
ann_model = bot_ann_model(input_dim=len(
    tfidf_X_train[0]), num_classes=num_classes)
ann_model.fit(tfidf_X_train, tfidf_y_train, epochs=75, batch_size=5, verbose=1)

# Save the trained ANN model
ann_model.save('ann_model.h5')

# Train SVM model using Word2Vec embeddings
svm_model = bot_svm_model()
svm_model.fit(X_word2vec, tags_encoded)

# Save the trained SVM model
joblib.dump(svm_model, 'svm_model.pkl')

# Split data for training and testing with Bag-of-Words embeddings
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(
    X_bow, y, test_size=0.2, random_state=42)

# Define timesteps for LSTM and reshape data
timesteps = 1
input_shape = (timesteps, X_train_bow.shape[1])
X_train_bow = np.reshape(
    X_train_bow, (X_train_bow.shape[0], timesteps, X_train_bow.shape[1]))
X_test_bow = np.reshape(
    X_test_bow, (X_test_bow.shape[0], timesteps, X_test_bow.shape[1]))

# Train LSTM model
lstm_model = bot_lstm_model(input_shape=input_shape, num_classes=num_classes)
lstm_model.fit(X_train_bow, y_train_bow, epochs=75, batch_size=5, verbose=1)

# Save the trained LSTM model
lstm_model.save('lstm_model.h5')

# Save embedding models and other utilities
save_embedding_models(vectorizer, label_encoder,
                      tag_responses, word2vec_model, bow_model=bow_model)
