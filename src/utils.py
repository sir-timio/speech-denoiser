import io
import numpy as np
import librosa
from urllib.request import urlopen
from pydub import AudioSegment
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)


def plot_wave(wav: np.ndarray, sr: int) -> plt.figure:
    """plot waveform of wav audio data

    Args:
        wav (np.ndarray): audio data
        sr (int): sample rate

    Returns:
        plt.figure: waveform figure
    """
    _, ax = plt.subplots()
    librosa.display.waveshow(wav, sr=sr, axis="time", ax=ax)
    return plt.gcf()
