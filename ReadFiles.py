import os
import config
from music21 import converter, note, chord


def parse_midi(file_path):
    """解析MIDI文件，提取音符和和弦"""
    midi = converter.parse(file_path)
    notes = []
    for element in midi.flat:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


for midi_file in config.MIDIFILES:
    notes = parse_midi(midi_file)
    print(f"Parsed {midi_file}: {notes}")