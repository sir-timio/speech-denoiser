import torch
import numpy as np
import librosa
import librosa.display

import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt

from src.metrics import compute_STOI, compute_PESQ
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
# from augment import Augment


class LitModel(pl.LightningModule):
    def __init__(self, config, model, loss_fn, logger):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.sample_rate = config.trainer.sample_rate
        self.debug_interval = config.trainer.debug_interval
        self.logger = logger

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=1e-06,
        )

    def forward(self, inp):
        out = self.model(inp)
        return out

    def training_step(self, batch, batch_ind):
        noisy, clean, _ = batch
        enhanced = self.forward(noisy)
        loss = self.loss_fn(clean.float(), enhanced.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_ind):
        noisy, clean, _ = batch
        enhanced = self.forward(noisy)
        val_loss = self.loss_fn(clean.float(), enhanced.float())
        self.log('val_loss', val_loss)

        # stoi and pesq require too much time...
        enhanced = enhanced[0].cpu().numpy().reshape(-1)
        clean = clean[0].cpu().numpy().reshape(-1)
        noisy = noisy[0].cpu().numpy().reshape(-1)
        
        # stoi_c_e = compute_STOI(clean, enhanced, sr=self.sample_rate)
        stoi_c_e = 0
        self.log('stoi_clean&enhanced', stoi_c_e)
        # pesq_c_e = compute_PESQ(clean, enhanced, sr=self.sample_rate)
        pesq_c_e = 0
        self.log('pesq_clean&enhanced', stoi_c_e)

        return {
            "stoi_clean&enhanced": stoi_c_e,
            "pesq_сlean&enhanced": pesq_c_e,
            "val_loss": val_loss
        }
        

    @staticmethod
    def _transform_pesq_range(pesq_score):
        """transform PESQ range. From [-0.5 ~ 4.5] to [0 ~ 1]."""
        return (pesq_score + 0.5) / 5

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def write_audio_samples(self, noisy, enhanced, clean, epoch, name):
        self.writer.add_audio(
            f"Audio_{name}_Noisy",
            noisy,
            epoch,
            sr=self.sample_rate,
        )
        self.writer.add_audio(
            f"Audio_{name}_Enhanced",
            enhanced,
            epoch,
            sr=self.sample_rate,
        )
        self.writer.add_audio(
            f"Audio_{name}_Clean", clean, epoch, sr=self.sample_rate
        )

    def visualize_waveform(self, noisy_mix, enhanced, clean, epoch, name):
        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([noisy_mix, enhanced, clean]):
            ax[j].set_title(
                "mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                    np.mean(y), np.std(y), np.max(y), np.min(y)
                )
            )
            librosa.display.waveplot(y, sr=self.sample_rate, ax=ax[j])
        plt.tight_layout()
        self.writer.add_figure(f"Waveform_{name}", fig, epoch)

    def visualize_spectrogram(self, noisy_mix, enhanced, clean, epoch, name):
        noisy_mag, _ = librosa.magphase(
            librosa.stft(noisy_mix, n_fft=320, hop_length=160, win_length=320)
        )
        enhanced_mag, _ = librosa.magphase(
            librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320)
        )
        clean_mag, _ = librosa.magphase(
            librosa.stft(clean, n_fft=320, hop_length=160, win_length=320)
        )

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate(
            [
                noisy_mag,
                enhanced_mag,
                clean_mag,
            ]
        ):
            axes[k].set_title(
                f"mean: {np.mean(mag):.3f}, "
                f"std: {np.std(mag):.3f}, "
                f"max: {np.max(mag):.3f}, "
                f"min: {np.min(mag):.3f}"
            )
            librosa.display.specshow(
                librosa.amplitude_to_db(mag),
                cmap="magma",
                y_axis="linear",
                ax=axes[k],
                sr=self.sample_rate,
            )
        plt.tight_layout()
        self.writer.add_figure(f"Spectrogram_{name}", fig, epoch)