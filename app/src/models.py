import os

import torch
import whisper

from .demucs import Demucs


def load_denoiser(ckpt_path="weights/demucs.ckpt"):
    state = torch.load(ckpt_path, map_location="cpu")
    model = Demucs(**state["args"])
    model.load_state_dict(state["weights"])
    return model.eval()


def load_transcriber(version="base"):
    """load transcriber model
        https://github.com/openai/whisper
    Returns:
        model: transcriber
    """
    assert version in whisper.available_models()

    return whisper.load_model(version)
