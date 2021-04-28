"""
This file contains functions for working with MIDI files.
"""

import pretty_midi



def load_midi(path: str):
    """
    This function loads a MIDI file into a more useful format.
    Inputs: path - the path to the file to load
    """

    # Load the file
    midi_file = pretty_midi.PrettyMIDI(path)

    # Isolate piano roll
    piano_roll = midi_file.instruments[0].get_piano_roll()

    return piano_roll



if __name__ == '__main__':
    # Test file
    path = './maestro-v3.0.0/2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.midi'

    # Load midi
    out = load_midi(path)
