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
    version = os.getenv("DENOISER_V")
    assert version in ['dns64', 'dns48']
    return eval(f"pretrained.{version}().cpu()")


def load_transcriber():
    """load transcriber model
        https://github.com/openai/whisper
    Returns:
        model: transcriber
    """
    version = os.getenv("WHISPER_V")
    assert version in whisper.available_models()
    return whisper.load_model(version)
