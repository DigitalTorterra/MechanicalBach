import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle


# import data
path = 'maestro-v3.0.0-midi/maestro-v3.0.0/'

metadata = pd.read_csv(path + 'maestro-v3.0.0.csv')

notes = []


nSongs = len(metadata.index)

# import data from list
listFile = 'list.txt'
with open(listFile, 'rb') as f:
    notes = pickle.load(f)


sequence_length = 100

pitchnames = sorted(set(item for item in notes))

# number of different pitches (options for output)
n_vocab = len(pitchnames)

# dict that converts notes to an int
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# setup network input and output
network_input = []
network_output = []

# creates sequences
for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in sequence_in])
    network_output.append(note_to_int[sequence_out])

# amount of input data
n_patterns = len(network_input)

# resize data to work in LSTM
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(n_vocab)

# put output in np array
network_output = np.array(network_output)


# build model
model = Sequential()
model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(n_vocab, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

# creates checkpoint (so can end training partway through)
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    
callbacks_list = [checkpoint]     

# train model
model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)








