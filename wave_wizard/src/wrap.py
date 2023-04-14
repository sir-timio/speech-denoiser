import torch
import numpy as np
import librosa
import librosa.display

import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt

from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    SignalNoiseRatio,
)
# from augment import Augment


class LitModel(pl.LightningModule):
    def __init__(self, config, model, loss_fn, writer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.stoi = ShortTimeObjectiveIntelligibility(config.trainer.sample_rate, extended=False)
        self.pesq = PerceptualEvaluationSpeechQuality(config.trainer.sample_rate, mode="wb")
        self.snr = SignalNoiseRatio()
        self.writer = writer

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=1e-06,
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_ind):
        noisy, clean, _ = batch
        enhanced = self.forward(noisy)
        loss = self.loss_fn(clean, enhanced)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_ind):
        noisy, clean, _ = batch
        enhanced = self.forward(noisy)
        loss = self.loss_fn(clean, enhanced)
        self.log('val_loss', loss)
        return loss
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        noisy, clean, name = batch
        enhanced = self.forward(noisy)
        return dict(noisy=noisy, clean=clean, enhanced=enhanced), name
    
    
    def test_step(self, batch, batch_ind):
        noisy, clean, name = batch
        enhanced = self.forward(noisy)
        
        self.stoi(enhanced, clean)
        self.pesq(enhanced, clean)
        self.snr(enhanced, clean)
        
        self.log('stoi', self.stoi)
        self.log('pesq', self.pesq)
        self.log('snr', self.snr)
