import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MetricCallback(Callback):
    def __init__(self, test_loader, every_n_epochs=1):
        super().__init__()
        self.test_loader = test_loader
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (
            trainer.current_epoch != 0
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            for i, batch in enumerate(self.test_loader):
                batch = pl_module._apply_batch_transfer_handler(batch)
                pl_module.test_step(batch, i)


class AudioVisualizationCallback(Callback):
    def __init__(self, test_loader, writer, every_n_epochs=1):
        super().__init__()
        self.test_loader = test_loader
        self.sr = self.test_loader.dataset.sr
        self.writer = writer
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (
            trainer.current_epoch != 0
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            for i, batch in enumerate(self.test_loader):
                batch = pl_module._apply_batch_transfer_handler(batch)
                out, name = pl_module.predict_step(batch, i)
                name = name[0]
                self.write_audio(out, name, trainer.current_epoch)
                self.write_wave_plot(out, name, trainer.current_epoch)
                self.write_mel_plot(out, name, trainer.current_epoch)

    def write_audio(self, out, name, epoch):
        for k, tensor in out.items():
            self.writer.add_audio(f"{k}_{name}.wav", tensor, epoch, self.sr)

    def write_wave_plot(self, out, name, epoch, figsize=(10, 10)):
        fig, ax = plt.subplots(len(out), 1, figsize=figsize)
        for i, (k, tensor) in enumerate(out.items()):
            y = tensor.detach().cpu().numpy().reshape(-1)
            ax[i].set_title(f"{k}_{name}.wav")
            librosa.display.waveshow(y, sr=self.sr, ax=ax[i])
        plt.tight_layout()
        self.writer.add_figure(f'waveform_{name.split("_")[-1]}', fig, epoch)

    def write_mel_plot(self, out, name, epoch, figsize=(10, 10)):
        fig, ax = plt.subplots(len(out), 1, figsize=figsize)
        for i, (k, tensor) in enumerate(out.items()):
            y = tensor.detach().cpu().numpy().reshape(-1)
            ax[i].set_title(f"{k}_{name}.wav")
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr)
            librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), ax=ax[i])
        plt.tight_layout()
        self.writer.add_figure(f'melspec_{name.split("_")[-1]}', fig, epoch)
