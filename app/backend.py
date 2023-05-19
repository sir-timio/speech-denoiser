import io
import os
import sys

sys.path.append("../app")
import base64
import io
import uuid
from os.path import join as join_path

import librosa
import numpy as np
import requests
import soundfile as sf
import torchaudio
import uvicorn
from audio_recorder_streamlit import audio_recorder
from fastapi import FastAPI, File, Response, UploadFile
from matplotlib import pyplot as plt
from pydub import AudioSegment

from src.engine import cold_run, denoise, transcribe
from src.models import load_denoiser, load_transcriber
from src.utils import load_config, parse_args, plot_wave


def ogg_to_wav(file) -> bytes:
    """Telegram отправляет голосовые сообщения в формате ogg
    с кодеком OPUS. soundfile поддерживает ogg,
    но только с кодеком Vorbis, а не OPUS."""
    audio = AudioSegment.from_ogg(file)
    byte_io = io.BytesIO()
    audio.export(byte_io, format="wav")
    byte_io.seek(0)
    return byte_io


def np_to_ogg(waveform, sr):
    # convert numpy array to audio file
    audio = AudioSegment(
        waveform.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )

    byte_io = io.BytesIO()
    audio.export(byte_io, format="ogg")
    byte_io.seek(0)
    return byte_io


app = FastAPI()


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    converted_audio = ogg_to_wav(io.BytesIO(await file.read()))
    waveform, sr = sf.read(converted_audio, dtype="int16")

    # do some
    audio_bytes = np_to_ogg(waveform, sr)
    return {
        "voice": base64.b64encode(audio_bytes.getvalue()).decode("utf-8"),
        "text": "",
    }


if __name__ == "__main__":
    config = load_config(parse_args())

    if not os.path.exists(config["storage"]):
        os.makedirs(config["storage"])

    denoiser = load_denoiser(ckpt_path=config["denoiser_ckpt_path"])
    transcriber = load_transcriber(version=config["whisper_version"])
    uvicorn.run("backend:app", host="0.0.0.0", port=30025, workers=1, reload=False)
