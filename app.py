from flask import Flask, render_template, request, json, jsonify
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, converter, musicxml
import numpy as np
import os,base64, io
from pydub import AudioSegment
import time
from pathlib import Path
import subprocess

app = Flask(__name__)



# Load the LSTM model and other necessary data
lstm_model = load_model('LSTM_MODEL_MAIN.h5') 

with open(r'data.json') as f:
    data = json.load(f)

Note_Count = 250
length = data['length']
L_symb = data['L_symb']
reverse_mapping = data['reverse_mapping']
X_seed = np.array(data['X_seed'])  # Convert list back to numpy array

def chords_n_notes(snippet):
    melody = []
    offset = 0  # Incremental
    for i in snippet:
        # If it is a chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".")  # Separating the notes in a chord
            notes = []

            for j in chord_notes:
                inst_note = int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)

            chord_snip = chord.Chord(notes)
            chord_snip.offset = offset
            melody.append(chord_snip)
        # If it is a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            melody.append(note_snip)

        # Increase offset each iteration so that notes do not stack
        offset += 1
    
    print(melody)

    melody_stream = stream.Stream(melody)
    return melody_stream


def Melody_Generator(Note_Count, model, length, L_symb, reverse_mapping):
    seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
    Music = []
    Notes_Generated = []

    for i in range(Note_Count):
        seed = seed.reshape(1, length, 1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0  # diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index // float(L_symb)
        # Convert index_N to an integer or chord representation
        if isinstance(index_N, (int, float)):
            Notes_Generated.append(str(int(index_N)))
        else:
            chord_notes = reverse_mapping[str(int(index_N))].split(".")  # Get the notes in the chord
            Notes_Generated.extend([str(int(note)) for note in chord_notes])
            
        Music = [reverse_mapping[str(char)] for char in Notes_Generated]
        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]

    # Convert the generated music to a music stream
    Melody = chords_n_notes(Music)
    
    return Music, Melody

@app.route('/')
def index():
    # Use the loaded values as needed in your route function
    return render_template('firstpage.html', Note_Count=Note_Count, length=length, L_symb=L_symb, reverse_mapping=reverse_mapping, X_seed=X_seed)

@app.route('/generate_music', methods=['POST'])
def generate_music_endpoint():
    musician = request.json['musician']
    generated_music, music_stream = Melody_Generator(Note_Count, lstm_model, length, L_symb, reverse_mapping)
    
    music_stream.write('mid', 'static/output.mid')
    return jsonify({'generated_music': 'success'})

if __name__ == '__main__':
    app.run(debug=True,port =5000)