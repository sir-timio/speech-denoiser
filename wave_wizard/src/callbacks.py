import librosa
from inference import enhance
from pytorch_lightning.callbacks import Callback

class VisualizationCallback(Callback):
    def __init__(self, noisy_audio_files, every_n_epochs=1, sr=16000):
        super().__init__()
        for file in noisy_audio_files:
            signal, _ = librosa.load(file, sr=sr)
            
        self.input_audio = [
            
        ]