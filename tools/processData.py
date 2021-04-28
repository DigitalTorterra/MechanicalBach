import numpy as np
import pandas as pd
from music21 import converter, note, chord
import pickle


# import data
path = 'maestro-v3.0.0-midi/maestro-v3.0.0/'

metadata = pd.read_csv(path + 'maestro-v3.0.0.csv')

notes = []


nSongs = len(metadata.index)
# to do: add in offset
for i, filename in enumerate(metadata['midi_filename']):
    file = path + filename
    midi = converter.parse(file)

    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    
    if (i%10 == 0):
        print(f'{i + 1} songs done out of {nSongs}')

with open('list2.txt', 'wb') as fp:
    pickle.dump(notes, fp)