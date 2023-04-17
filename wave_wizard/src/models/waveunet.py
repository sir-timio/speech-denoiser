import torch
from torch import nn
from torch.nn import functional as F
from .util import rescale_conv, rescale_module


class WaveUnet(nn.Module):
    def __init__(
        self,
        chin=1,
        chout=1,
        hidden=24,
        depth=5,
        normalize=True,
        rescale=0.1,
        floor=1e-3,
    ):
        """WaveUnet

        Args:
            chin (int, optional): input channels. Defaults to 1.
            chout (int, optional): output channels. Defaults to 1.
            hidden (int, optional): number of initial hidden channels.. Defaults to 24.
            depth (int, optional): number of layers. Defaults to 5.
            normalize (bool, optional): if true, normalize the input. Defaults to True.
            rescale (float, optional): controls custom weight initialization. Defaults to 0.1.
            floor (_type_, optional): stability flooring when normalizing. Defaults to 1e-3.
        """
        super().__init__()

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.floor = floor
        self.normalize = normalize

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, 15, 1, 7),
                nn.BatchNorm1d(hidden),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, chout, 5, 1, 2),
                nn.BatchNorm1d(chout),
                nn.LeakyReLU(negative_slope=0.1),
            ]

            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden += hidden

        self.middle = nn.Sequential(nn.Conv1d(depth * hidden, depth * hidden, 15, 1, 7))
        if rescale:
            rescale_module(self, reference=rescale)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        x = mix

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
            x = x[:, :, ::2]

        # x = self.middle(x)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=True)
            x = x + skip[..., : x.shape[-1]]
            x = decode(x)

        return std * x


if __name__ == "__main__":
    model = WaveUnet()

    y = torch.rand((16, 1, 16384))

    print(model(y).shape)
