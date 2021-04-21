import numpy as np
import pandas as pd
from music21 import converter, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


# import data
path = 'maestro-v3.0.0-midi/maestro-v3.0.0/'

metadata = pd.read_csv(path + 'maestro-v3.0.0.csv')

notes = []

# print(metadata[metadata['year'] == 2018]['midi_filename'][0])

for filename in [metadata[metadata['year'] == 2018]['midi_filename'][0]]:
    file = path + filename
    midi = converter.parse(file)
    notes_to_parse = None

    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

sequence_length = 100

pitchnames = sorted(set(item for item in notes))

n_vocab = len(pitchnames)

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# setup network input and output
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(n_vocab)

network_output = to_categorical(network_output)
print(network_output.shape)

model = Sequential()
model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(n_vocab, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    
callbacks_list = [checkpoint]     
model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)








