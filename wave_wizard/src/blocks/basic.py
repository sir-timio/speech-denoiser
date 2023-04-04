import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

class BasicConv(nn.Module):
    def __init__(self,
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                dilation=1, groups=1, bias=True,
                activation=nn.PReLU()):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride, padding=padding, 
            dilation=dilation, groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation
        

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
    
class BasicDeConv(nn.Module):
    def __init__(self,
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                dilation=1, groups=1, bias=True,
                activation=nn.PReLU()):
        super(BasicDeConv, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride, padding=padding, 
            dilation=dilation, groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation
        

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x