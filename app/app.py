import uuid
import os
from os.path import join as join_path
from src.models import load_denoiser
from src.utils import plot_wave
from src.engine import denoise, cold_run
import streamlit as st
import torchaudio
import librosa
from matplotlib import pyplot as plt
from audio_recorder_streamlit import audio_recorder


plt.rcParams["figure.figsize"] = (10, 5)


STORAGE_FOLDER = "storage"


def process_audio(audio_bytes, denoiser):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("Исходная аудиозапись")
        st.audio(audio_bytes, format="audio/wav")

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    file_path = join_path(STORAGE_FOLDER, file_name)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    wav, sr = librosa.load(file_path)

    with col1:
        st.pyplot(plot_wave(wav, sr))

    denoised_file_path = join_path(
        STORAGE_FOLDER, file_name.replace(uid, f"denoised_{uid}")
    )
    denoised_audio_tensor = denoise(denoiser, file_path)

    torchaudio.save(denoised_file_path, denoised_audio_tensor, denoiser.sample_rate)

    with col2:
        st.header("Очищенная аудиозапись")
        st.audio(open(denoised_file_path, "rb").read(), format="audio/wav")
        denoised_wav, sr = librosa.load(denoised_file_path)
        st.pyplot(plot_wave(denoised_wav, sr))


def main(denoiser):
    st.title("Audio denoiser")

    audio_recorder_input = audio_recorder(
        text="Запишите звук",
        pause_threshold=5.0,
        neutral_color="#1ceb6b",
    )
    uploaded_file = st.file_uploader(
        "или загрузите файл", type=["wav", "mp3", "mp4", "ogg"]
    )

    # audio recorder handler
    if audio_recorder_input:
        process_audio(audio_recorder_input, denoiser)

    # upload file handler
    if uploaded_file:
        process_audio(uploaded_file.read(), denoiser)


if __name__ == "__main__":
    if not os.path.exists(STORAGE_FOLDER):
        os.makedirs(STORAGE_FOLDER)

    main(load_denoiser())
