import sys
sys.path.append('..')

import os
import uuid
from flask import Flask, flash, request, redirect, send_file, jsonify

# model 
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import librosa
    

UPLOAD_FOLDER = 'files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '123456'

@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/save-record', methods=['POST'])
def save_record():    
    audio_data = request.files.get('data')
    
    if not audio_data:
        return jsonify({'success': False, 'message': 'No record found'}), 400
    
    uid = str(uuid.uuid4())
    fname = os.path.join(app.config['UPLOAD_FOLDER'], uid + ".wav")
    print(fname)
    
    audio_data.save(fname)
        
    assert MODEL, 'load model first'
    
    wav, sr = librosa.load(fname)
    wav = convert_audio(torch.tensor(wav[None,:]), sr, MODEL.sample_rate, MODEL.chin)
    with torch.no_grad():
        denoised = MODEL(wav[None])[0]
    
    denoised_fname = os.path.join(app.config['UPLOAD_FOLDER'], "denoised_" + uid + ".wav")
    torchaudio.save(denoised_fname, denoised, MODEL.sample_rate)
    
    return send_file(denoised_fname, as_attachment=True)
    # return jsonify({'success': True, 'message': 'Record saved'}), 200


@app.route('/denoised_audio/<file>')
def denoise(file: str) -> str:
    assert MODEL is None, 'load model first'
    wav, sr = librosa.load(file)
    wav = convert_audio(torch.tensor(wav[None,:]), sr, MODEL.sample_rate, MODEL.chin)
    with torch.no_grad():
        denoised = MODEL(wav[None])[0]
    
    return denoised


if __name__ == '__main__':
    MODEL = pretrained.dns64().cpu()
    app.run(debug=False)
