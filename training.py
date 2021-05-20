# External imports
import argparse
import json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

# Internal imports
import data
import models
import gan

# Constants
MODEL_LIST = ['LSTM', 'transformer', 'GAN']
DATA_MODES = ['Numeric']

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('-m', '--model_type', help=f'Model Type', choices=MODEL_LIST, required=True)
    parser.add_argument('-n', '--name', help='Experiment name', required=True)
    parser.add_argument('-l', '--loss_function', help='Loss function to use', default='sparse_categorical_crossentropy')
    parser.add_argument('-o', '--optimizer', help='Optimizer to use', default='adam')
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('-d', '--data_mode', help=f'How to encode the MIDI data', choices=DATA_MODES, default='Numeric')
    parser.add_argument('-s', '--seq_len', help='Length of input sequence to model', type=int, default=50)
    parser.add_argument('-p', '--data_path', help='Path to training data', default='./data/train.pkl')
    parser.add_argument('-w', '--weights_path', help='Path to directory to store weights', default='./weights/')
    parser.add_argument('-t', '--tensorboard_dir', help='TensorBoard log path', default='./logs/')

    # LSTM-Specific Arguments
    parser.add_argument('--lstm_size', help='Size of LSTM layers', type=int, default=256)
    parser.add_argument('--lstm_num_layers', help='Number of LSTM layers', type=int, default=3)
    parser.add_argument('--lstm_dropout_prob', help='LSTM dropout probability', type=float)
    parser.add_argument('--lstm_num_hidden_dense', help='Number of hidden dense layers', type=int, default=2)
    parser.add_argument('--lstm_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256)
    parser.add_argument('--lstm_hidden_dense_activation', help='Activation for hidden dense layers', default='relu')
    parser.add_argument('--lstm_embedding_size', help='Included embedding layer size', type=int)

    # Transformer-Specific Arguments
    parser.add_argument('--embed_dim', help='Embedding size for each token', type=int, default=32)
    parser.add_argument('--num_heads', help='Number of attention heads', type=int, default=2)
    parser.add_argument('--ff_dim', help='Hidden layer size in feed forward network inside transformer', type=int, default=32)
    parser.add_argument('--transformer_dropout_prob', help='transformer dropout probability', type=float)
    parser.add_argument('--transformer_num_hidden_dense', help='Number of hidden dense layers', type=int, default=2)
    parser.add_argument('--transformer_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256)
    parser.add_argument('--transformer_hidden_dense_activation', help='Activation for hidden dense layers', default='relu')

    # GAN-Specific Arguments
    parser.add_argument('--gan_latent_dims', help='Size of latent space', type=int, default=100)
    parser.add_argument('--gan_num_dense_layers', help='Number of dense layers', type=int, default=1)
    parser.add_argument('--gan_dense_hidden_size', help='Size of hidden layers', type=int, default=1000)
    parser.add_argument('--gan_starting_num_channels', help='Number of channels in convolution', type=int, default=64)
    parser.add_argument('--gan_activation', help='GAN activation function')
    parser.add_argument('--gan_num_hidden_conv_layers', help='Number of hidden convolutional layers', type=int, default=3)
    parser.add_argument('--gan_hidden_conv_num_channels', help='Number of channels for hidden convolutional channels', type=int, default=256)
    parser.add_argument('--gan_dropout_prob', help='GAN dropout probability', type=float)



    # Parse arguments
    args = parser.parse_args()

    # Initialize dataset
    if args.data_mode == 'Numeric' and args.model_type != 'GAN':
        normalize_in = args.model_type == 'LSTM' and args.lstm_embedding_size == None
        dataset = data.MIDINumericDataset(path=args.data_path, sequence_len=args.seq_len, normalize_in=normalize_in, onehot_out=False)
        out_shape = dataset.n_vocab

    elif args.data_mode == 'Numeric' and args.model_type == 'GAN':
        dataset = data.MIDIOnehotDataset(path=args.data_path, sequence_len=args.seq_len)


    # Preprocess data
    network_input, network_output = dataset.get_data()
    pitchnames, n_vocab = dataset.pitchnames, dataset.n_vocab

    # Initialize tensorboard
    if args.tensorboard_dir != None:
        update_freq = 'batch' if args.model_type != 'GAN' else None
        tensorboard = TensorBoard(log_dir=f'{args.tensorboard_dir}{args.name}',
                                  update_freq=update_freq,
                                  profile_batch=0)
        tensorboard._should_write_train_graph = False

    # Create model
    if args.model_type == 'LSTM':
        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, args.seq_len, 1))
        network_input = np.array(network_input) / float(n_vocab)
        network_output = np.array(network_output)
        in_shape = (network_input.shape[1], network_input.shape[2])

        if args.lstm_embedding_size != None:
            embedding_in_size = out_shape
            embedding_out_size = args.lstm_embedding_size
            embedding_seq_len = args.seq_len
        else:
            embedding_in_size = None
            embedding_out_size = None
            embedding_seq_len = None

        print('c')

        model, hparams = models.create_lstm(input_shape = in_shape,
                                            out_size = n_vocab,
                                            embedding_in_size = embedding_in_size,
                                            embedding_out_size = embedding_out_size,
                                            embedding_seq_len = embedding_seq_len,
                                            lstm_size = args.lstm_size,
                                            num_lstm_layers = args.lstm_num_layers,
                                            dropout_prob = args.lstm_dropout_prob,
                                            num_hidden_dense = args.lstm_num_hidden_dense,
                                            hidden_dense_size = args.lstm_hidden_dense_size,
                                            hidden_dense_activation = args.lstm_hidden_dense_activation,
                                            loss_function = args.loss_function,
                                            optimizer = args.optimizer)
        print('d')
        
        # Setup training checkpoints
        filepath = f'{args.weights_path}{args.name}.hdf5'
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        if args.tensorboard_dir != None:
            callbacks_list.append(tensorboard)

        # Save args
        hparam_path = f'{args.weights_path}{args.name}.json'
        with open(hparam_path, 'w') as f:
            json.dump(hparams, f)

        print(model.summary())

        print('e')

        # Train the model
        model.fit(network_input, network_output, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list)
    
    elif args.model_type == 'transformer':
        model, hparams = models.create_transformer(input_length = args.seq_len,
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
        
        # Save args
        hparam_path = f'{args.weights_path}{args.name}.json'
        with open(hparam_path, 'w') as f:
            json.dump(hparams, f)


        # Load tensorboard
        callbacks = [tensorboard] if args.tensorboard_dir != None else None


        # Train the model
        model.fit(network_input, network_output, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
        filepath = f'{args.weights_path}{args.name}.hdf5'
        model.save_weights(filepath)


    elif args.model_type == 'GAN':
        gen, dis, gan_model, hparams = models.create_gan(latent_dim = args.gan_latent_dims,
                                            num_dense_layers = args.gan_num_dense_layers,
                                            dense_hidden_size = args.gan_dense_hidden_size,
                                            starting_num_channels = args.gan_starting_num_channels,
                                            activation = args.gan_activation,
                                            num_hidden_conv_layers = args.gan_num_hidden_conv_layers,
                                            hidden_conv_num_channels = args.gan_hidden_conv_num_channels,
                                            n_vocab = dataset.n_vocab,
                                            input_length = dataset.sequence_len,
                                            disc_drop_prob = args.gan_dropout_prob,
                                            loss_function = args.loss_function,
                                            optimizer = args.optimizer)

        # Save args
        hparam_path = f'{args.weights_path}{args.name}.json'
        with open(hparam_path, 'w') as f:
            json.dump(hparams, f)
        #
        # Load tensorboard
        callbacks = [tensorboard] if args.tensorboard_dir != None else None

        # Train model
        gan.train_gan(gen, dis, gan_model, dataset, latent_dims=args.gan_latent_dims, n_epochs=args.epochs,
                      batch_size=args.batch_size, base_path=args.weights_path, test_name=args.name, callbacks=callbacks)
