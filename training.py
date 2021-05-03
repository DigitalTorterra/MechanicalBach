# External imports
import argparse
import json
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Internal imports
import data
import models

# Constants
MODEL_LIST = ['LSTM', 'transformer']
DATA_MODES = ['Numeric']

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('-m', '--model_type', help=f'Model Type', choices=MODEL_LIST, required=True)
    parser.add_argument('-n', '--name', help='Experiment name', required=True)
    parser.add_argument('-l', '--loss_function', help='Loss function to use', default='categorical_crossentropy')
    parser.add_argument('-o', '--optimizer', help='Optimizer to use', default='rmsprop')
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('-d', '--data_mode', help=f'How to encode the MIDI data', choices=DATA_MODES, default='Numeric')
    parser.add_argument('-s', '--seq_len', help='Length of input sequence to model', type=int, default=50)
    parser.add_argument('-p', '--data_path', help='Path to training data', default='./data/train.pkl')
    parser.add_argument('-w', '--weights_path', help='Path to directory to store weights', default='./weights/')

    # LSTM-Specific Arguments
    parser.add_argument('--lstm_size', help='Size of LSTM layers', type=int, default=512)
    parser.add_argument('--lstm_num_layers', help='Number of LSTM layers', type=int, default=3)
    parser.add_argument('--lstm_dropout_prob', help='LSTM dropout probability', type=float)
    parser.add_argument('--lstm_num_hidden_dense', help='Number of hidden dense layers', type=int, default=1)
    parser.add_argument('--lstm_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256)
    parser.add_argument('--lstm_hidden_dense_activation', help='Activation for hidden dense layers', default='relu')

    # Transformer-Specific Arguments
    parser.add_argument('--embed_dim', help='Embedding size for each token', type=int, default=32)
    parser.add_argument('--num_heads', help='Number of attention heads', type=int, default=2)
    parser.add_argument('--ff_dim', help='Hidden layer size in feed forward network inside transformer', type=int, default=32)
    parser.add_argument('--transformer_dropout_prob', help='transformer dropout probability', type=float)
    parser.add_argument('--transformer_num_hidden_dense', help='Number of hidden dense layers', type=int, default=1)
    parser.add_argument('--transformer_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256)
    parser.add_argument('--transformer_hidden_dense_activation', help='Activation for hidden dense layers', default='relu')

    # Parse arguments
    args = parser.parse_args()


    # Initialize dataset
    if args.data_mode == 'Numeric':
        dataset = data.MIDINumericDataset(path=args.data_path, sequence_len=args.seq_len)
        out_shape = dataset.n_vocab


    # Preprocess data
    network_input, network_output = dataset.get_data()
    pitchnames, n_vocab = dataset.get_pitch_metadata(dataset.notes)

    # Create model
    if args.model_type == 'LSTM':
        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, args.seq_len, 1))
        network_input = np.array(network_input) / float(n_vocab)
        network_output = np.array(network_output)
        in_shape = (network_input.shape[1], network_input.shape[2])

        model = models.create_lstm(in_shape,
                                   n_vocab,
                                   lstm_size = args.lstm_size,
                                   num_lstm_layers = args.lstm_num_layers,
                                   dropout_prob = args.lstm_dropout_prob,
                                   num_hidden_dense = args.lstm_num_hidden_dense,
                                   hidden_dense_size = args.lstm_hidden_dense_size,
                                   hidden_dense_activation = args.lstm_hidden_dense_activation,
                                   loss_function = args.loss_function,
                                   optimizer = args.optimizer)
        
        # Setup training checkpoints
        filepath = f'{args.weights_path}{args.name}' + '-{epoch:02d}-{loss:.4f}.hdf5'
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        # Save args
        with open(argpath, 'w') as f:
            json.dump(hparams, f)

        print(model.summary())

        # Train the model
        model.fit(network_input, network_output, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list)
    
    elif args.model_type == 'transformer':
        model = models.create_transformer(input_length = args.seq_len,
                                          n_vocab=n_vocab,
                                          embed_dim=args.embed_dim,
                                          num_heads=args.num_heads,
                                          ff_dim=args.ff_dim,
                                          dropout_prob = args.transformer_dropout_prob,
                                          num_hidden_dense = args.transformer_num_hidden_dense,
                                          hidden_dense_size = args.transformer_hidden_dense_size,
                                          hidden_dense_activation = args.transformer_hidden_dense_activation,
                                          loss_function = args.loss_function,
                                          optimizer = args.optimizer)

        # Train the model
        model.fit(network_input, network_output, epochs=args.epochs, batch_size=args.batch_size)
        filepath = f'{args.weights_path}{args.name}'
        model.save_weights(filepath)

