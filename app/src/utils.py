import numpy as np
import librosa
from matplotlib import pyplot as plt

import argparse
import yaml
import argparse
from addict import Dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        default="configs/web.yaml",
        help="YAML configuration file",
    )
    return parser.parse_args()


def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return Dict(config)


def plot_wave(wav: np.ndarray, sr: int, figsize: tuple = (20, 10)) -> plt.figure:
    """plot waveform of wav audio data

    Args:
        wav (np.ndarray): audio data
        sr (int): sample rate

    Returns:
        plt.figure: waveform figure
    """
    _, ax = plt.subplots(figsize=figsize)
    librosa.display.waveshow(wav, sr=sr, axis="time", ax=ax)
    return plt.gcf()
