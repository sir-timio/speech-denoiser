import torch
from torch import nn
from torch.nn import functional as F


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class WaveUnet(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
    """
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
        
        self.middle = nn.Sequential(
            nn.Conv1d(depth * hidden, depth * hidden, 15, 1, 7)
        )
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