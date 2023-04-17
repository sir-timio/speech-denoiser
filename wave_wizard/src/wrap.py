import torch

import pytorch_lightning as pl
import torch.optim as optim

from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    SignalNoiseRatio,
)


class LitModel(pl.LightningModule):
    def __init__(self, config, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.sample_len = config["dataset"]["sample_len"]
        self.stoi = ShortTimeObjectiveIntelligibility(
            config["trainer"]["sample_rate"], extended=False
        )
        self.pesq = PerceptualEvaluationSpeechQuality(
            config["trainer"]["sample_rate"], mode="wb"
        )
        self.snr = SignalNoiseRatio()

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config["optim"]["lr"],
            weight_decay=1e-06,
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def adaptive_forward(self, x):
        """only for one batched test data, specially for waveunet implementation"""
        if x.size(-1) % self.sample_len != 0:
            padded_length = self.sample_len - (x.size(-1) % self.sample_len)
            x = torch.cat(
                [
                    x,
                    self._apply_batch_transfer_handler(
                        torch.zeros(size=(1, 1, padded_length))
                    ),
                ],
                dim=-1,
            )
        x = torch.reshape(x, shape=(-1, 1, self.sample_len))

        enhanced_chunks = self.model(x)
        enhanced = enhanced_chunks.reshape(-1)
        if padded_length != 0:
            enhanced = enhanced[:-padded_length]

        enhanced = torch.reshape(enhanced, shape=(1, 1, -1))
        return enhanced

    def training_step(self, batch, batch_ind):
        noisy, clean, _ = batch
        enhanced = self.forward(noisy)
        loss = self.loss_fn(clean, enhanced)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_ind):
        noisy, clean, _ = batch
        enhanced = self.forward(noisy)
        loss = self.loss_fn(clean, enhanced)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        noisy, clean, name = batch
        enhanced = self.adaptive_forward(noisy)
        return dict(noisy=noisy, clean=clean, enhanced=enhanced), name

    def test_step(self, batch, batch_ind):
        noisy, clean, name = batch
        enhanced = self.adaptive_forward(noisy)

        self.stoi(enhanced, clean)
        self.pesq(enhanced, clean)
        self.snr(enhanced, clean)

        self.log("stoi", self.stoi)
        self.log("pesq", self.pesq)
        self.log("snr", self.snr)
