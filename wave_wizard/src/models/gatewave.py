import math
import sys

from torch import nn
from torch.nn import functional as F

sys.path.append("../..")
from src.blocks import *  # noqa

from .util import rescale_conv, rescale_module


class BGRU(nn.Module):
    def __init__(self, dim, layers=1, bi=True):
        super().__init__()
        klass = nn.GRU
        self.rnn = klass(
            bidirectional=bi,
            num_layers=layers,
            hidden_size=dim,
            input_size=dim,
        )
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.rnn(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


class GateWave(nn.Module):
    def __init__(
        self,
        depth=3,
        scale=1.618,
        init_hidden=32,
        kernel_size=7,
        stride=1,
        padding=2,
        encoder_class="GatedConv",
        decoder_class="GatedDeConv",
        normalize=True,
        rescale=None,
    ):
        """GateWave

        Args:
            depth (int, optional): number of layers. Defaults to 3.
            scale (float, optional): _description_. Defaults to 1.618.
            init_hidden (int, optional): number of initial hidden channels. Defaults to 32.
            kernel_size (int, optional): kernel size for each layer. Defaults to 7.
            stride (int, optional): stride for each layer. Defaults to 1.
            padding (int, optional): padding for each layer. Defaults to 2.
            encoder_class (str, optional): GatedConv or BacisConv. Defaults to "GatedConv".
            decoder_class (str, optional): GatedDeConv or BasicDeConv. Defaults to "GatedDeConv".
            normalize (bool, optional): if true, normalize input. Defaults to True.
        """
        super(GateWave, self).__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        in_channels = 1
        out_channels = 1
        self.normalize = normalize
        self.floor = 1e-3
        encoders = []
        decoders = []
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        hidden = init_hidden
        in_ch = in_channels
        for i in range(depth):
            encoder = eval(encoder_class)(
                in_channels, hidden, kernel_size, stride, padding
            )
            self.encoder.append(encoder)
            encoders.append(encoder)

            decoder = eval(decoder_class)(
                hidden, out_channels, kernel_size, stride, padding
            )
            decoders.append(decoders)

            self.decoder.insert(0, decoder)
            out_channels = hidden
            in_channels = hidden
            hidden = int(hidden * scale)

        self.rnn = BGRU(in_channels, bi=True)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        return int(length)

    def forward(self, x):
        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = x / (self.floor + std)
        else:
            std = 1
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., : x.shape[-1]]
            x = decode(x)

        x = x[..., :length]
        return std * x


if __name__ == "__main__":
    import torch

    model = GateWave()
    print(model)
    y = torch.rand((16, 1, 16384))
    print(model(y).shape)
