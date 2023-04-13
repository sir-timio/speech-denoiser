import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
import math

from ..blocks import BasicConv, BasicDeConv, GatedConv, GatedDeConv


class LitGateWave(pl.LightningModule):
    def __init__(self,
                 depth=3, scale=2, init_hidden=32,
                 kernel_size=7, stride=1, padding=2,
                 encoder_class=GatedConv,
                 decoder_class=GatedDeConv):
        super().__init__()
        self.save_hyperparameters()
                
        self.loss_fn = nn.L1Loss()
        in_channels = 1
        out_channels = 1
        encoders = []
        decoders = []
        
        hidden = init_hidden
        for i in range(depth):
            encoder = encoder_class(in_channels, hidden, kernel_size, stride, padding)
            encoders.append(encoder)
            
            decoder = decoder_class(hidden, out_channels, kernel_size, stride, padding)
            decoders.append(decoder)
            
            out_channels = hidden
            in_channels = hidden
            hidden = int(hidden * scale)

        self.encoder = nn.Sequential(*encoders)
        self.decoder = nn.Sequential(*decoders[::-1])
    
    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        for idx in range(self.hparams.depth):
            length = math.ceil((length - self.hparams.kernel_size) / self.hparams.stride) + 1
            length = max(length, 1)
        for idx in range(self.hparams.depth):
            length = (length - 1) * self.hparams.stride + self.hparams.kernel_size
        return int(length)

    def pad(self, x):
        length = x.shape[-1]
        return F.pad(x, (0, self.valid_length(length) - length))
    
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        noisy = self.pad(noisy)
        clean = self.pad(clean)
        denoised = self(noisy)
        loss = self.loss_fn(denoised, clean)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        noisy = self.pad(noisy)
        clean = self.pad(clean)
        denoised = self(noisy)
        loss = self.loss_fn(denoised, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = self.loss_fn(denoised, clean)
        self.log('test_loss', loss)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)