import sys

sys.path.append("../app")
import uuid
import os
from src.demucs import Demucs
from os.path import join as join_path
from src.models import load_denoiser, load_transcriber
from src.utils import plot_wave
from src.engine import denoise, transcribe, cold_run
from src.utils import load_config, parse_args
import streamlit as st
import torchaudio
import librosa
from matplotlib import pyplot as plt
from audio_recorder_streamlit import audio_recorder

plt.rcParams["figure.figsize"] = (10, 5)


def process_audio(storage, audio_bytes, denoiser, transcriber):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("Исходная аудиозапись")
        st.audio(audio_bytes, format="audio/wav")

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    file_path = join_path(storage, file_name)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    wav, sr = librosa.load(file_path)

    origin_text = transcribe(transcriber, file_path) or "(пусто)"

    with col1:
        st.pyplot(plot_wave(wav, sr))

    denoised_file_path = join_path(storage, file_name.replace(uid, f"denoised_{uid}"))
    denoised_audio_tensor = denoise(denoiser, file_path)

    torchaudio.save(denoised_file_path, denoised_audio_tensor, denoiser.sample_rate)

    with col2:
        st.header("Очищенная аудиозапись")
        st.audio(open(denoised_file_path, "rb").read(), format="audio/wav")
        denoised_wav, sr = librosa.load(denoised_file_path)
        st.pyplot(plot_wave(denoised_wav, sr))

    st.header(f"Транскрипт:\n{origin_text}")


def main(storage, denoiser, transcriber):
    st.title("Audio denoiser and trascriber 🤫")
    audio_recorder_input = audio_recorder(
        text="Запишите звук",
        pause_threshold=5.0,
        neutral_color="#1ceb6b",
    )
    uploaded_file = st.file_uploader(
        "или загрузите файл", type=["wav", "mp3", "mp4", "ogg"]
    )

    if audio_recorder_input:
        process_audio(storage, audio_recorder_input, denoiser, transcriber)

    # upload file handler
    if uploaded_file:
        process_audio(storage, uploaded_file.read(), denoiser, transcriber)


if __name__ == "__main__":
    config = load_config(parse_args())

    if not os.path.exists(config["storage"]):
        os.makedirs(config["storage"])

    denoiser = load_denoiser(ckpt_path=config["denoiser_ckpt_path"])
    transcriber = load_transcriber(version=config["whisper_version"])
    main(config["storage"], denoiser, transcriber)
