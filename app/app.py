import uuid
import os
from os.path import join as join_path
from src.models import load_denoiser, load_transcriber
from src.utils import plot_wave
from src.engine import denoise, transcribe, cold_run
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import torchaudio
import librosa
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)
from dotenv import load_dotenv


load_dotenv()


STORAGE_FOLDER = os.getenv("STORAGE_FOLDER")


def process_audio(audio_bytes, denoiser, transcriber):
    col1, col2 = st.columns(2, gap="large")
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

    denoised_file_path = join_path(STORAGE_FOLDER,
                                   file_name.replace(uid, f"denoised_{uid}"))
    denoised_audio_tensor = denoise(denoiser, file_path)

    torchaudio.save(denoised_file_path, denoised_audio_tensor,
                    denoiser.sample_rate)

    with col2:
        st.header("Очищенная аудиозапись")
        st.audio(open(denoised_file_path, 'rb').read(), format="audio/wav")
        denoised_wav, sr = librosa.load(denoised_file_path)
        st.pyplot(plot_wave(denoised_wav, sr))

    _, text = transcribe(transcriber, denoised_file_path)
    text = text or "(пусто)"
    st.header(f"Транскрипт:\n{text}")


def main(denoiser, transcriber):
    st.title("Audio denoiser and trascriber 🤫")
    audio_recorder_input = audio_recorder(text="Запишите звук",
                                          pause_threshold=5.0,
                                          neutral_color="#1ceb6b",                                          
                                          )
    uploaded_file = st.file_uploader("или загрузите файл",
                                     type=["wav", "mp3", "mp4", "ogg"])
    
    # audio recorder handler
    if audio_recorder_input:
        process_audio(audio_recorder_input, denoiser, transcriber)

    # upload file handler
    if uploaded_file:
        process_audio(uploaded_file.read(), denoiser, transcriber)


if __name__ == "__main__":
    if not os.path.exists(STORAGE_FOLDER):
        os.makedirs(STORAGE_FOLDER)
        
    denoiser = load_denoiser()
    transcriber = load_transcriber()
    # cold_run([denoiser, transcriber], [denoise, transcribe])
    main(denoiser, transcriber)
