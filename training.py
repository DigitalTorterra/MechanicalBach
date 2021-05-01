# External imports
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint

# Internal imports
import data
import models

# Constants
MODEL_LIST = ['LSTM']
DATA_MODES = ['Numeric']

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

    # Create model
    if args.model_type == 'LSTM':
        model = models.create_lstm(network_input[0].shape,
                                   out_shape,
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

    # Train the model
    model.fit(network_input, network_output, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list)
