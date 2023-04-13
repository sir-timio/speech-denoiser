from torch import nn
from torch.nn import functional as F
import math
from src.blocks import *

class GateWave(nn.Module):
    def __init__(self,
                 depth=3, scale=2, init_hidden=32,
                 kernel_size=7, stride=1, padding=2,
                 encoder_class=BasicConv,
                 decoder_class=BasicDeConv):
        super(GateWave, self).__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        in_channels = 1
        out_channels = 1
        encoders = []
        decoders = []
        
        hidden = init_hidden
        in_ch = in_channels
        for i in range(depth):
            
            encoder = eval(encoder_class)(in_channels, hidden, kernel_size, stride, padding)
            encoders.append(encoder)
            
            decoder = eval(decoder_class)(hidden, out_channels, kernel_size, stride, padding)
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
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        return int(length)

    def forward(self, x):
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        latent = self.encoder(x)
        output = self.decoder(latent)
        output = output[..., :length]
        return output
