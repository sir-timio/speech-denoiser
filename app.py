from typing import Dict, Tuple
import uuid
from os.path import join as join_path

import streamlit as st
from audio_recorder_streamlit import audio_recorder

import torch
import torchaudio
import librosa
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)

# denoiser
from denoiser import pretrained
from denoiser.dsp import convert_audio

# transcriber
import whisper


STORAGE_FOLDER = 'files'


def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    return plt.gcf()

def load_denoiser():
    return pretrained.dns64().cpu()


def denoise(model: torch.nn.Module, file_path: str) -> str:
    """denoising func

    Args:
        model (torch.nn.Module): denoising model
        file_path (str): path to local audio file

    Returns:
        str: 
    """
    wav, sr = librosa.load(file_path)
    wav = convert_audio(torch.tensor(wav[None,:]), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    return denoised

def load_transcriber():
    return whisper.load_model('base')

def transcribe(model, file_path: str) -> Tuple:
    result = model.transcribe(file_path)
    
    return result["language"], result["text"]

def main():
    st.title("Audio denoiser")
    audio_bytes = audio_recorder(text="Запишите звук", pause_threshold=5.0)
    st.file_uploader("или загрузите файл")
    col1, col2 = st.columns(2, gap="large")

    denoiser = load_denoiser()
    transcriber = load_transcriber()
    
    if audio_bytes:
        with col1:
            st.header("Исходная аудиозапись")
            st.audio(audio_bytes, format="audio/wav")
        
        uid = str(uuid.uuid4())
        file_name = uid + '.wav'
        
        file_path = join_path(STORAGE_FOLDER, file_name)
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        wav, sr = librosa.load(file_path)
        
        with col1:
            st.pyplot(plot_wave(wav, sr))
        
        denoised_file_path = join_path(STORAGE_FOLDER, file_name.replace(uid, f"denoised_{uid}"))
        denoised_audio_tensor = denoise(denoiser, file_path)
        
        torchaudio.save(denoised_file_path, denoised_audio_tensor, denoiser.sample_rate)        
        
        with col2:
            st.header("Очищенная аудиозапись")
            st.audio(open(denoised_file_path, 'rb').read(), format="audio/wav")
            denoised_wav, sr = librosa.load(denoised_file_path)
            st.pyplot(plot_wave(denoised_wav, sr))
    
        _, text = transcribe(transcriber, denoised_file_path)
        st.header(f"Транскрипт:\n{text}")        
        
        
        

if __name__ == "__main__":
    main()
