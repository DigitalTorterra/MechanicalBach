# External libraries
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Reshape, Conv2DTranspose, Conv2D, Flatten
from tensorflow.keras import layers
from utils import get_starting_size

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


def load_from_dict(in_dict):
    """
    This function loads an arbitrary model from a dict
    """
    # Get model name
    model_type = in_dict.pop('model_type')

    # Load LSTM
    if model_type == 'lstm':
        return create_lstm(**in_dict)
    elif model_type == 'transformer':
        return create_transformer(**in_dict)
    elif model_type == 'gan':
        return create_gan(**in_dict)



def create_lstm(input_shape: Tuple[int, int] = (100, 1),
                out_size: int = 100,
                embedding_in_size: int = None,
                embedding_out_size: int = None,
                embedding_seq_len: int = None,
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
        'embedding_in_size': embedding_in_size,
        'embedding_out_size': embedding_out_size,
        'embedding_seq_len': embedding_seq_len,
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

    if embedding_in_size != None and embedding_out_size != None and embedding_seq_len != None:
        model.add(Embedding(embedding_in_size, embedding_out_size, input_length=embedding_seq_len))
        model.add(LSTM(lstm_size, return_sequences=True))
    else:
        # Add initial LSTM layer
        model.add(LSTM(lstm_size, input_shape=input_shape, return_sequences=(num_lstm_layers != 1)))

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

    return model, args

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

        # Save arguments
    args = {
        'model_type': 'transformer',
        'input_length': input_length,
        'n_vocab': n_vocab,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
        'dropout_prob': dropout_prob,
        'num_hidden_dense': num_hidden_dense,
        'hidden_dense_size': hidden_dense_size,
        'hidden_dense_activation': hidden_dense_activation,
        'loss_function': loss_function,
        'optimizer': optimizer,
    }

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

    return model, args

def create_gan(latent_dim: int = 100,
               num_dense_layers: int = 1,
               dense_hidden_size: int = 1000,
               starting_num_channels: int = 128,
               activation: str = 'relu',
               num_hidden_conv_layers: int = 3,
               hidden_conv_num_channels: int = 256,
               n_vocab: int = 100,
               input_length: int = 100,
               disc_drop_prob: float = None,
               loss_function: str = 'binary_crossentropy',
               optimizer: str = 'rmsprop'):
    """
    "Image" size: (input_length, n_vocab, 1)
    """
    args = {
        'model_type': 'gan',
        'latent_dim': latent_dim,
        'num_dense_layers': num_dense_layers,
        'dense_hidden_size': dense_hidden_size,
        'starting_num_channels': starting_num_channels,
        'activation': activation,
        'num_hidden_conv_layers': num_hidden_conv_layers,
        'hidden_conv_num_channels': hidden_conv_num_channels,
        'n_vocab': n_vocab,
        'input_length': input_length,
        'disc_drop_prob': disc_drop_prob,
        'loss_function': loss_function,
        'optimizer': optimizer,
    }

    # Calculate starting size
    starting_image_size, num_doublings = get_starting_size(num_hidden_conv_layers, n_vocab, input_length)

    # Initialize generator
    generator = Sequential()

    # Add dense layers
    for i in range(num_dense_layers):
        out_size = starting_image_size[0]*starting_image_size[1]*starting_num_channels if i + 1 == num_dense_layers else dense_hidden_size
        if i == 0:
            generator.add(Dense(out_size, input_dim=latent_dim, activation=activation))
        else:
            generator.add(Dense(out_size, activation=activation))

    # Add reshape layer
    generator.add(Reshape((*starting_image_size, starting_num_channels)))

    # Add intermediate convolutional layers
    for i in range(num_hidden_conv_layers):
        if i < num_doublings:
            generator.add(Conv2DTranspose(hidden_conv_num_channels, (4, 4), strides=(2,2), padding='same', activation=activation))
        else:
            generator.add(Conv2DTranspose(hidden_conv_num_channels, (2, 2), strides=(1,1), padding='same', activation=activation))

    # Add final convolutional layer
    generator.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    generator.compile(loss=loss_function, optimizer=optimizer)

    # Initialize discriminator
    discriminator = Sequential()

    # Add convolutional layers
    for i in range(num_hidden_conv_layers+1):
        if i == 0:
            discriminator.add(Conv2D(hidden_conv_num_channels, (3, 3), padding='same', input_shape=(input_length, n_vocab, 1)))
        else:
            discriminator.add(Conv2D(hidden_conv_num_channels, (3, 3), padding='same', activation=activation))
        if disc_drop_prob != None:
            discriminator.add(Dropout(disc_drop_prob))

    # Add the flattening
    discriminator.add(Flatten())

    # Add dense layers
    for i in range(num_dense_layers):
        if i + 1 == num_dense_layers:
            discriminator.add(Dense(1, activation='sigmoid'))
        else:
            discriminator.add(Dense(dense_hidden_size, activation=activation))
            if disc_drop_prob != None:
                discriminator.add(Dropout(disc_drop_prob))

    discriminator.compile(loss=loss_function, optimizer=optimizer)

    # Create combined model
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss=loss_function, optimizer=optimizer)
    discriminator.trainable = True

    return generator, discriminator, gan, args
