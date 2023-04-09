from pesq import pesq
from pystoi import stoi

def compute_STOI(clean_signal, noisy_signal, sr=22050):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def compute_PESQ(clean_signal, noisy_signal, sr=22050):
    return pesq(sr, clean_signal, noisy_signal, "wb")
