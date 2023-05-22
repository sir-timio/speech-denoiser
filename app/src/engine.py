from typing import Tuple

import librosa
import torch
import whisper

from .demucs import convert_audio


def denoise(model, file_path: str) -> torch.Tensor:
    """denoising with encoder decoder model

    Args:
        model: model from https://github.com/facebookresearch/denoiser
        file_path (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    wav, sr = librosa.load(file_path)
    wav = convert_audio(torch.tensor(wav[None, :]), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    return denoised


def transcribe(model, file_path: str) -> Tuple:
    """transcribe audiofile with openai whisper

    Args:
        model: model from  https://github.com/openai/whisper
        file_path (str): _description_

    Returns:
        Tuple: _description_
    """
    audio = whisper.pad_or_trim(whisper.load_audio(file_path))

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    result = model.transcribe(file_path)
    return result["text"]


def cold_run(models, funcs: callable, file_path: str = "cold_run.wav"):
    """cold run for models

    Args:
        models:
        funcs: _description_
        file_path (str, optional): dummy data. Defaults to 'cold_run.wav'.
    """
    print("Start cold run")
    for model, func in zip(models, funcs):
        func(model, file_path=file_path)
    print("Cold run ended")
