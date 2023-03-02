import os
from denoiser import pretrained
import whisper
from dotenv import load_dotenv


load_dotenv()


def load_denoiser():
    """load denoising model
        https://github.com/facebookresearch/denoiser
    Returns:
        model: denoiser
    """
    return pretrained.dns64().cpu()


def load_transcriber():
    """load transcriber model
        https://github.com/openai/whisper
    Returns:
        model: transcriber
    """
    return whisper.load_model(os.getenv("WHISPER_V"))
