import uuid
import os
from os.path import join as join_path
from app.src.models import load_transcriber
from src.models import load_denoiser
from src.utils import plot_wave
from src.engine import denoise, cold_run
import streamlit as st
import torchaudio
import librosa
from matplotlib import pyplot as plt
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import argparse
import yaml
import argparse


plt.rcParams["figure.figsize"] = (10, 5)

parser = argparse.ArgumentParser(description="Process some integers.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        default="configs/train_config.yaml",
        help="YAML configuration file",
    )
    return parser.parse_args()


def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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
    st.title("Audio denoiser and trascriber 🤫")

    webrtc_ctx = webrtc_streamer(
        key="speech denoising",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )
    status_indicator = st.empty()

    uploaded_file = st.file_uploader(
        "или загрузите файл", type=["wav", "mp3", "mp4", "ogg"]
    )
    # if not webrtc_ctx.state.playing:
    #     return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        st.audio(audio_frames.tobytes())
        process_audio(audio_frames.tobytes(), denoiser)

    print(audio_frames)
    # upload file handler
    if uploaded_file:
        process_audio(uploaded_file.read(), denoiser)


if __name__ == "__main__":
    config = load_config(parse_args())

    if not os.path.exists(config["storage"]):
        os.makedirs(config["storage"])

    denoiser = load_denoiser(ckpt_path=config["denoiser_ckpt_path"])
    transcriber = load_transcriber(version=config["whisper_version"])
    main(load_denoiser())
