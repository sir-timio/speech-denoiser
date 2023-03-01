from distutils.log import debug
import os
import uuid
from flask import Flask, flash, request, redirect, send_file, jsonify
from base64 import b64decode
import librosa

UPLOAD_FOLDER = 'files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '123'

@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/save-record', methods=['POST'])
def save_record():    
    audio_data = request.files.get('data')
    
    if not audio_data:
        return jsonify({'success': False, 'message': 'No record found'}), 400
    
    fname = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + ".wav")
    audio_data.save(fname)
    
    return jsonify({'success': True, 'message': 'Record saved'}), 200


@app.route('/play-audio')
def play_audio():
    audio_file = '/Users/timur/web-denoiser/files/1d9c88ea-0768-4362-bc43-e63ebbf4fad9.mp3'
    if os.path.isfile(audio_file):
        return send_file(audio_file, mimetype='audio/wav')
    else:
        return 'Audio file not found'

def denoise(model, audiofile):
    pass

if __name__ == '__main__':
    app.run(debug=True)
