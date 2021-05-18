# MechanicalBach

This project is a neural network which composes piano music. The three models weree trained by the following dataset: https://magenta.tensorflow.org/datasets/maestro#download

## Background 

This project compares three different models' (a GAN, an LSTM, and a Transformer) performance in being trained and ability to compose original music. Each neural network is trained on the Maestro dataset, and the trained models can then 'compose' original pieces of classical piano. 

## Running the code 

As the code is parametrized, only `training.py` and `generate.py` need to be run and the choice of models and hyperparameters can be chosen from the command line. 

### Training.py 

`python training.py` 

#### General arguments

Required arguments: 
*  '-m', '--model_type', help=f'Model Type', choices=MODEL_LIST, required=True 
*  '-n', '--name', help='Experiment name', required=True
  
Other Command Line Arguments (Hyperparameters): 
*  '-l', '--loss_function', help='Loss function to use', default='sparse_categorical_crossentropy'
*  '-o', '--optimizer', help='Optimizer to use', default='adam'
*  '-b', '--batch_size', help='Batch size', type=int, default=64
*  '-e', '--epochs', help='Number of epochs', type=int, default=10
*  '-d', '--data_mode', help=f'How to encode the MIDI data', choices=DATA_MODES, default='Numeric'
*  '-s', '--seq_len', help='Length of input sequence to model', type=int, default=50
*  '-p', '--data_path', help='Path to training data', default='./data/train.pkl'
*  '-w', '--weights_path', help='Path to directory to store weights', default='./weights/'
*  ('-t', '--tensorboard_dir', help='TensorBoard log path', default='./logs/'
  
  ##### LSTM-Specific Arguments
*  '--lstm_size', help='Size of LSTM layers', type=int, default=256
*  '--lstm_num_layers', help='Number of LSTM layers', type=int, default=3
*  '--lstm_dropout_prob', help='LSTM dropout probability', type=float
*  '--lstm_num_hidden_dense', help='Number of hidden dense layers', type=int, default=2
*  '--lstm_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256
*  '--lstm_hidden_dense_activation', help='Activation for hidden dense layers', default='relu'
*  '--lstm_embedding_size', help='Included embedding layer size', type=int
    
 #### Transformer-Specific Arguments
*  '--embed_dim', help='Embedding size for each token', type=int, default=32
*  '--num_heads', help='Number of attention heads', type=int, default=2
*  '--ff_dim', help='Hidden layer size in feed forward network inside transformer', type=int, default=32
*  '--transformer_dropout_prob', help='transformer dropout probability', type=float
*  '--transformer_num_hidden_dense', help='Number of hidden dense layers', type=int, default=2
*  '--transformer_hidden_dense_size', help='Size of hidden denses layers', type=int, default=256
*  '--transformer_hidden_dense_activation', help='Activation for hidden dense layers', default='relu'

#### GAN-Specific Arguments
*  '--gan_latent_dims', help='Size of latent space', type=int, default=100
*  '--gan_num_dense_layers', help='Number of dense layers', type=int, default=1
*  '--gan_dense_hidden_size', help='Size of hidden layers', type=int, default=1000
*  '--gan_starting_num_channels', help='Number of channels in convolution', type=int, default=64
*  '--gan_activation', help='GAN activation function'
*  '--gan_num_hidden_conv_layers', help='Number of hidden convolutional layers', type=int, default=3
*  '--gan_hidden_conv_num_channels', help='Number of channels for hidden convolutional channels', type=int, default=256
*  '--gan_dropout_prob', help='GAN dropout probability', type=float

### Generate.py 

`python generate.py`

### 

