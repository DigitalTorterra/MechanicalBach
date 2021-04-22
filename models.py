# External libraries
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout





def create_lstm(input_shape: Tuple[int, int],
                out_size: int,
                lstm_size: int =  512,
                num_lstm_layers: int = 3,
                dropout_prob: float = None,
                num_hidden_dense: int = 1,
                hidden_dense_size: int = 256,
                hidden_dense_activation: str = 'relu',
                loss_function: str = 'categorical_crossentropy',
                optimizer: str = 'rmsprop'):

    # Initialize model
    model = Sequential()

    # Add initial LSTM layer
    model.add(LSTM(lstm_size, input_shape=input_shape, return_sequences=True))

    # Add subsequent LSTM layers and dropout if necessary
    for _ in range(num_lstm_layers-1):
        if dropout_prob != None:
            model.add(Dropout(dropout_prob))

        model.add(LSTM(lstm_size, return_sequences=True))

    # Add dense layers
    for _ in range(num_hidden_dense):
        model.add(Dense(hidden_dense_size, activation='relu'))

        if dropout_prob != None:
            model.add(Dropout(dropout_prob))

    # Add final layer
    model.add(Dense(out_size, activation='softmax'))

    # Compile model
    model.compile(loss=loss_function, optimizer=optimizer)

    return model
