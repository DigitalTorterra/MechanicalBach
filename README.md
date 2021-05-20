# MechanicalBach

This project is a neural network which composes piano music. The three models were trained by the following dataset: https://magenta.tensorflow.org/datasets/maestro#download

## Background 

This project compares three different models' (a GAN, an LSTM, and a Transformer) performance in being trained and ability to compose original music. Each neural network is trained on the Maestro dataset, and the trained models can then 'compose' original pieces of classical piano. 

## Explanation of Files

### Main Directory

* `data.py`: Contains files for parsing the dataset into a form usable by Keras.
* `gan.py`: Contains extra tools for training the GAN
* `generate.py`: Allows you to use pre-trained models to generate new music
* `models.py`: Contains descriptions of the LSTM, GAN, and Transformer models
* `training.py`: Contains a script that allows you to train models with arbitrary hyperparameters
* `utils.py`: Contains miscellaneous helper functions

### `data` Directory

This directory contains the parsed dataset.

* `list.txt`: A pickle file containing the entire parsed dataset
* `test.pkl`: A picle file containing the test set
* `train.pkl`: A picle file containing the training set
* `val.pkl`: A picle file containing the validation set

### `tools` Directory

This directory contains the scripts used to create the parsed dataset.

* `processData.py`: This file contained the initial data parsing script
* `process_split.py`: This is an improved data parsing file that also includes multithreading and train/test split.
