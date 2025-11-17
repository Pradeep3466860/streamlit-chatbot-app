from keras.layers import LSTM, Dense, Dropout  # type: ignore
from keras.models import Sequential  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.svm import SVC


"""
Artificial Neural Networks are computational models inspired by the human brain, 
consisting of interconnected nodes (neurons) organized in layers. 
They are used for various tasks, including classification, regression, and pattern recognition.

Support Vector Machines are supervised learning models used for classification and regression tasks. 
They work by finding the hyperplane that best separates data points of different classes in a high-dimensional space.

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN)
designed to remember long-term dependencies and mitigate the vanishing gradient problem. 
It uses a gating mechanism to control the flow of information, allowing it to maintain and update memory over long sequences.

"""


def bot_ann_model(input_dim, num_classes):
    """
    Create and compile a simple Artificial Neural Network (ANN) model.
    
    Parameters:
    - input_dim: int, the number of features in the input data.
    - num_classes: int, the number of classes for classification.

    Returns:
    - model: A compiled Keras Sequential model.
    """
    model = Sequential()

    # Input layer with 128 neurons and ReLU activation function
    model.add(Dense(128, input_shape=(input_dim,), activation='relu'))

    # Dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Hidden layer with 64 neurons and ReLU activation function
    model.add(Dense(64, activation='relu'))

    # Dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Output layer with softmax activation function for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model


def bot_svm_model():
    """
    Create a Support Vector Machine (SVM) model.
    
    Returns:
    - model: A SVC model with probability estimates enabled.
    """
    return SVC(probability=True)



def bot_lstm_model(input_shape, num_classes):
    """
    Create and compile a simple LSTM model.
    
    Parameters:
    - input_shape: tuple, the shape of the input data (timesteps, features).
    - num_classes: int, the number of classes for classification.

    Returns:
    - model: A compiled Keras Sequential model.
    """
    model = Sequential()

    # LSTM layer with 100 units
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))

    # Dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Another LSTM layer with 50 units
    model.add(LSTM(50))

    # Dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Output layer with softmax activation function for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model



