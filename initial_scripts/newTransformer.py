import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# External imports
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
import keras_transformer


# Internal imports
import data
import models

# Constants
MODEL_LIST = ['LSTM']
DATA_MODES = ['Numeric']

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


if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('-m', '--model_type', help=f'Model Type', choices=MODEL_LIST, required=True)
    parser.add_argument('-n', '--name', help='Experiment name', required=True)
    parser.add_argument('-l', '--loss_function', help='Loss function to use', default='sparse_categorical_crossentropy')
    parser.add_argument('-o', '--optimizer', help='Optimizer to use', default='rmsprop')
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-d', '--data_mode', help=f'How to encode the MIDI data', choices=DATA_MODES, default='Numeric')
    parser.add_argument('-s', '--seq_len', help='Length of input sequence to model', type=int, default=100)
    parser.add_argument('-p', '--data_path', help='Path to training data', default='./data/train.pkl')
    parser.add_argument('-w', '--weights_path', help='Path to directory to store weights', default='./weights/')

    # LSTM-Specific Arguments
    parser.add_argument('--lstm_size', help='Size of LSTM layers', type=int, default=512)
    parser.add_argument('--lstm_num_layers', help='Number of LSTM layers', type=int, default=3)
    parser.add_argument('--lstm_dropout_prob', help='LSTM dropout probability', type=float)
    parser.add_argument('--lstm_num_hidden_dense', help='Number of hidden dense layers', type=int, default=1)
    parser.add_argument('--lstm_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256)
    parser.add_argument('--lstm_hidden_dense_activation', help='Activation for hidden dense layers', default='relu')

    # Parse arguments
    args = parser.parse_args()


    # Initialize dataset
    if args.data_mode == 'Numeric':
        dataset = data.MIDINumericDataset(path=args.data_path, sequence_len=args.seq_len)
        out_shape = 1


    # Preprocess data
    network_input, network_output = dataset.get_data()

    pitchnames, n_vocab = dataset.get_pitch_metadata(dataset.notes)
    seqlength = args.seq_len


    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(seqlength,))
    embedding_layer = TokenAndPositionEmbedding(seqlength, n_vocab, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_vocab, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Setup training checkpoints
    filepath = f'{args.weights_path}{args.name}' + '-{epoch:02d}-{loss:.4f}.hdf5'

    # Train the model
    model.fit(network_input, network_output, epochs=args.epochs, batch_size=args.batch_size)

    model.save_weights('transWeights')
