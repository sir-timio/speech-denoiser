import io
import os
import sys

sys.path.append("../app")
import uuid
from os.path import join as join_path

import librosa
import numpy as np
import soundfile as sf
import torchaudio
import uvicorn
from audio_recorder_streamlit import audio_recorder
from fastapi import FastAPI, File, UploadFile
from matplotlib import pyplot as plt

from src.engine import cold_run, denoise, transcribe
from src.models import load_denoiser, load_transcriber
from src.utils import load_config, parse_args, plot_wave

app = FastAPI()


@app.get("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    contents = await file.read()
    waveform, sr = sf.read(file=io.BytesIO(contents), dtype="float32")
    byte_io = io.BytesIO()
    sf.write(byte_io, waveform, sr, format="WAV")
    audio_bytes = byte_io.getvalue()
    return audio_bytes


@app.get("/process_filename")
async def process_filename(file: UploadFile = File(...)):
    return 200


if __name__ == "__main__":
    config = load_config(parse_args())

    if not os.path.exists(config["storage"]):
        os.makedirs(config["storage"])

    denoiser = load_denoiser(ckpt_path=config["denoiser_ckpt_path"])
    transcriber = load_transcriber(version=config["whisper_version"])
    uvicorn.run("backend:app", host="0.0.0.0", port=30025, workers=1, reload=False)
