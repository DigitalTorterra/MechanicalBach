# External libraries
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def load_from_dict(in_dict):
    """
    This function loads an arbitrary model from a dict
    """
    # Get model name
    model_type = in_dict.pop(model_type)

    # Load LSTM
    if model_type == 'LSTM':
        create_lstm(**in_dict)



def create_lstm(input_shape: Tuple[int, int] = (100, 1),
                out_size: int = 100,
                lstm_size: int =  512,
                num_lstm_layers: int = 3,
                dropout_prob: float = None,
                num_hidden_dense: int = 1,
                hidden_dense_size: int = 256,
                hidden_dense_activation: str = 'relu',
                loss_function: str = 'sparse_categorical_crossentropy',
                optimizer: str = 'rmsprop'):

    # Save arguments
    args = {
        'model_type': 'lstm',
        'input_shape': input_shape,
        'out_size': out_size,
        'lstm_size': lstm_size,
        'num_lstm_layers': num_lstm_layers,
        'dropout_prob': dropout_prob,
        'num_hidden_dense': num_hidden_dense,
        'hidden_dense_size': hidden_dense_size,
        'hidden_dense_activation': hidden_dense_activation,
        'loss_function': loss_function,
        'optimizer': optimizer,
    }

    # Initialize model
    model = Sequential()

    # Add initial LSTM layer
    model.add(LSTM(lstm_size, input_shape=input_shape, return_sequences=True))

    # Add subsequent LSTM layers and dropout if necessary
    for i in range(num_lstm_layers-1):
        if dropout_prob != None:
            model.add(Dropout(dropout_prob))

        return_sequences = i != num_lstm_layers-2
        model.add(LSTM(lstm_size, return_sequences=return_sequences))

    # Add dense layers
    for _ in range(num_hidden_dense):
        model.add(Dense(hidden_dense_size, activation='relu'))

        if dropout_prob != None:
            model.add(Dropout(dropout_prob))

    # Add final layer
    model.add(Dense(out_size, activation='softmax'))

    # Compile model
    model.compile(loss=loss_function, optimizer=optimizer)

    return model, args
