import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import nltk

# Download required NLTK resources (important for Streamlit Cloud)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import base64
import os

st.set_page_config(page_title="Teesside FAQ Chatbot", layout="centered")

# Load the label encoder, responses, and vectorizers
label_encoder = joblib.load('label_encoder.pkl')
tag_responses = joblib.load('tag_responses.pkl')
vectorizer = joblib.load('vectorizer.pkl')
bow_vectorizer = joblib.load('bow_model.pkl')

# Load the models
ann_model = load_model('ann_model.h5')
lstm_model = load_model('lstm_model.h5')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def process_text(text):
    """Clean and preprocess text."""
    token = word_tokenize(text)
    token = [word.lower() for word in token if word.isalpha()]
    token = [word for word in token if word not in stop_words]
    token = [lemmatize(word) for word in token]
    return ' '.join(token)


def lemmatize(word):
    try:
        return lemmatizer.lemmatize(word)
    except:
        return word


def predict_response(text, model_type):
    processed_text = process_text(text)

    if model_type == 'ANN':
        text_vectorized = vectorizer.transform([processed_text]).toarray()
        probs = ann_model.predict(text_vectorized)
        predicted = np.argmax(probs, axis=1)
        tag = label_encoder.inverse_transform(predicted)[0]

    elif model_type == 'LSTM':
        text_vectorized = bow_vectorizer.transform([processed_text]).toarray()
        text_vectorized = np.reshape(text_vectorized, (1, 1, text_vectorized.shape[1]))
        probs = lstm_model.predict(text_vectorized)
        predicted = np.argmax(probs, axis=1)
        tag = label_encoder.inverse_transform(predicted)[0]

    else:
        return "Invalid model."

    responses = tag_responses.get(tag, ["Sorry, I don't understand."])
    return np.random.choice(responses)


def set_background(image_path):
    """Set background image (if available)."""
    if not os.path.exists(image_path):
        return  # Don't crash if image missing

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
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Load background
set_background('bg.jpg')

# Streamlit UI
st.title("Teesside FAQ Chatbot 🤖")
st.write("Ask me anything about Teesside University!")

model_type = st.radio("Choose model:", ["ANN", "LSTM"])
user_input = st.text_input("You:", "")

if user_input:
    response = predict_response(user_input, model_type)
    st.write("### Chatbot:", response)
