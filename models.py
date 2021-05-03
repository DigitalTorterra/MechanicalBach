# External libraries
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import layers

# Defining layers for Transformer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



def create_lstm(input_shape: Tuple[int, int],
                out_size: int,
                lstm_size: int =  512,
                num_lstm_layers: int = 3,
                dropout_prob: float = None,
                num_hidden_dense: int = 1,
                hidden_dense_size: int = 256,
                hidden_dense_activation: str = 'relu',
                loss_function: str = 'sparse_categorical_crossentropy',
                optimizer: str = 'rmsprop'):

    # Initialize model
    model = Sequential()

    # Add initial LSTM layer
    model.add(LSTM(lstm_size, input_shape=input_shape, return_sequences=True))

    # Add subsequent LSTM layers and dropout if necessary
    for i in range(num_lstm_layers-1):
        if dropout_prob != None:
            model.add(Dropout(dropout_prob))

        model.add(LSTM(lstm_size, return_sequences=(i != num_lstm_layers-2)))

    # Add dense layers
    for _ in range(num_hidden_dense):
        model.add(Dense(hidden_dense_size, activation=hidden_dense_activation))

        if dropout_prob != None:
            model.add(Dropout(dropout_prob))

    # Add final layer
    model.add(Dense(out_size, activation='softmax'))

    # Compile model
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

def create_transformer(input_length: int,
            n_vocab: int,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout_prob: float = None,
            num_hidden_dense: int = 1,
            hidden_dense_size: int = 256,
            hidden_dense_activation: str = 'relu',
            loss_function: str = 'sparse_categorical_crossentropy',
            optimizer: str = 'adam'):
    
    inputs = layers.Input(shape=(input_length,))
    embedding_layer = TokenAndPositionEmbedding(input_length, n_vocab, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Flatten()(x)

    # Add dense layers
    for _ in range(num_hidden_dense):
        if dropout_prob != None:
            x = layers.Dropout(dropout_prob)(x)
        
        x = layers.Dense(hidden_dense_size, activation=hidden_dense_activation)(x)

    # add final layer
    outputs = layers.Dense(n_vocab, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, loss=loss_function)

    return model
