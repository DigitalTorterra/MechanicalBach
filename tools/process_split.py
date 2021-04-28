import numpy as np
import pandas as pd
from music21 import converter, note, chord
import itertools
import pickle
import concurrent.futures

# Constants
SAVE_PATH   = 'data/'
BASE_PATH   = 'maestro-v3.0.0/'
NUM_THREADS = 8

# Functions
def extract_row(row):
    """
    This function extracts the notes from
    a row of data
    """

    # Get metadata from row
    split = row[1]['split']
    fname = row[1]['midi_filename']

    # Load MIDI file
    path = BASE_PATH + fname
    midi = converter.parse(path)
    notes_to_parse = midi.flat.notes

    # Parse notes/chords
    notes = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return split,notes


if __name__ == '__main__':
    # Import data
    metadata = pd.read_csv(BASE_PATH + 'maestro-v3.0.0.csv')

    # Create processpoolexecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        out = list(executor.map(extract_row, metadata.iterrows()))

    # Split into different lists
    train = [notes for (split, notes) in out if split == 'train']
    val   = [notes for (split, notes) in out if split == 'validation']
    test  = [notes for (split, notes) in out if split == 'test']

    # Unroll lists
    train = list(itertools.chain.from_iterable(train))
    val = list(itertools.chain.from_iterable(train))
    test = list(itertools.chain.from_iterable(train))

   

    # Write to pickle file
    with open(f'{SAVE_PATH}train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'{SAVE_PATH}val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(f'{SAVE_PATH}test.pkl', 'wb') as f:
        pickle.dump(test, f)



