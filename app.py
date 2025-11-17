import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import base64

# Load the label encoder, responses, and vectorizers
label_encoder = joblib.load('label_encoder.pkl')
tag_responses = joblib.load('tag_responses.pkl')
vectorizer = joblib.load('Embeddings/vectorizer.pkl')
bow_vectorizer = joblib.load('Embeddings/bow_model.pkl')

# Load the models
ann_model = load_model('Models/ann_model.h5')
lstm_model = load_model('Models/lstm_model.h5')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(text):
    """
    Process input text: tokenization, stopword removal, and lemmatization.
    """
    token = word_tokenize(text)
    token = [word.lower() for word in token if word.isalpha()]
    token = [word for word in token if word not in stop_words]
    token = [lemmatizer.lemmatize(word) for word in token]
    return ' '.join(token)

def predict_response(text, model_type):
    """
    Predict the response for a given input text using the selected model.
    """
    processed_text = process_text(text)

    if model_type == 'ANN':
        text_vectorized = vectorizer.transform([processed_text]).toarray()
        ann_probabilities = ann_model.predict(text_vectorized)
        ann_predicted_class = np.argmax(ann_probabilities, axis=1)
        tag = label_encoder.inverse_transform(ann_predicted_class)[0]

    elif model_type == 'LSTM':
        text_vectorized = bow_vectorizer.transform([processed_text]).toarray()
        text_vectorized = np.reshape(text_vectorized, (text_vectorized.shape[0], 1, text_vectorized.shape[1]))
        lstm_probabilities = lstm_model.predict(text_vectorized)
        lstm_predicted_class = np.argmax(lstm_probabilities, axis=1)
        tag = label_encoder.inverse_transform(lstm_predicted_class)[0]

    else:
        return "Invalid model type selected."

    responses = tag_responses.get(tag, ["Sorry, I don't understand."])
    return np.random.choice(responses)

def set_background(image_path):
    """
    Set the background image using base64 encoding.
    """
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background('image/bg.jpg')

# Streamlit App
st.title('Teesside FAQ Chatbot')

st.write("Hello! I'm here to assist you with your queries. How can I help you today?")

# Model selection
model_type = st.radio("Choose the model:", ("ANN", "LSTM"))

# User input
user_input = st.text_input("You:", "")

# Predict and display response
if user_input:
    response = predict_response(user_input, model_type)
    st.write(f"Chatbot: {response}")
