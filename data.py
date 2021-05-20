# External libraries
import pickle as pkl
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical


# Constants
DATA_PATH = './data/list.txt'
REF_PATH  = './data/list.txt'


class MIDIDataset:
    """
    This class is the base class for reading MIDI data
    """
    def __init__(self, path: str = DATA_PATH, ref_path: str = REF_PATH):

        # Save inputs
        self.path = path
        self.ref_path = ref_path

        # Load the pickled notes
        self.notes = self.load_pickle(path)
        self.ref_notes = self.load_pickle(ref_path)

        # Extract relevant metadata
        self.pitchnames, self.n_vocab = self.get_pitch_metadata(self.ref_notes)
        self.note_to_int = self.construct_note_dict(self.pitchnames)

        # Convert to NN form (placeholder)
        self.network_input, self.network_output = self.construct_data()


    def load_pickle(self, path: str):
        # This function reads in the pickle file
        with open(path, 'rb') as f:
            notes = pkl.load(f)

        return notes

    def get_pitch_metadata(self, notes):
        # This function returns a set of pitch-related metadata
        pitchnames = list(sorted(set(notes)))
        n_vocab = len(pitchnames)
        print(f'n_vocab: {n_vocab}')

        return pitchnames, n_vocab

    def construct_note_dict(self, pitchnames):
        # This function returns a mapping from notes to note number
        return {note:number for number, note in enumerate(pitchnames)}

    def construct_data(self):
        # Placeholder function
        return [], []

    def get_data(self):
        # Get data
        return self.network_input, self.network_output


class MIDINumericDataset(MIDIDataset):
    """
    This class extends `MIDIDataset` to represent each note as a scalar from
    0 to 1
    """
    def __init__(self, path: str = DATA_PATH, ref_path: str = REF_PATH, sequence_len: int = 100, normalize_in: bool = False, onehot_out: bool = True):
        # Save params
        self.sequence_len = sequence_len
        self.normalize_in = normalize_in
        self.onehot_out = onehot_out

        # Initialize base class
        super().__init__(path=path, ref_path=ref_path)


    def construct_data(self):
        # This function constructs the data by embedding it into the range [0,1]

        # Initialize outputs
        network_input  = []
        network_output = []

        # Create sequences
        for i in range(len(self.notes) - self.sequence_len):
            # Isolate the correct range
            sequence_in  = self.notes[i:i + self.sequence_len]
            sequence_out = self.notes[i + self.sequence_len]

            # Add to output
            network_input.append([self.note_to_int[n] for n in sequence_in])
            network_output.append(self.note_to_int[sequence_out])

        # Resize data- done in training b/c transformer does not need resized data
        # n_patterns = len(network_input)
        # network_input = np.reshape(network_input, (n_patterns, self.sequence_len, 1))

        # Normalize input- done in if statement below
        # network_input = np.array(network_input) / float(self.n_vocab)
        network_input = np.array(network_input)
        print(network_input.shape)

        network_output = np.array(network_output)
        if self.normalize_in:
            network_input = network_input / float(self.n_vocab)

        # Transform output
        if self.onehot_out:
            network_output = to_categorical(network_output, num_classes=self.n_vocab, dtype=network_input.dtype)

        return network_input, network_output


class MIDIOnehotDataset(MIDINumericDataset):
    """
    This class represents notes as one-hot vectors instead
    of embedding them into a float.
    """
    def construct_data(self):
        """
        This function constructs the dataset
        """
        network_input = [self.note_to_int[n] for n in self.notes]
        network_input = to_categorical(network_input, num_classes=self.n_vocab, dtype=np.float32)

        return network_input, []

    def get_idx(self, idx: int):
        # Returns a sequence starting from idx
        return self.network_input[idx:idx+self.sequence_len]
