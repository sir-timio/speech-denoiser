from torch import nn

class GatedConv(nn.Module):
    
    def __init__(self,
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                dilation=1, groups=1, bias=True,
                batch_norm=True,
                activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, padding=padding, 
                dilation=dilation, groups=groups,
                bias=bias
            )
        self.mask_conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, padding=padding, 
                dilation=dilation, groups=groups,
                bias=bias
            )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
    
    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv(input)
        mask = self.mask_conv(input)
        x = x * self.gated(mask)
        x = self.batch_norm(x)
        return x


class GatedDeConv(nn.Module):
    
    def __init__(self,
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                dilation=1, groups=1, bias=True,
                batch_norm=True,
                activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.deconv = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, padding=padding, 
                dilation=dilation, groups=groups,
                bias=bias
            )
        self.mask_deconv = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, padding=padding, 
                dilation=dilation, groups=groups,
                bias=bias
            )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight)
    
    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.deconv(input)
        mask = self.mask_deconv(input)
        x = x * self.gated(mask)
        x = self.batch_norm(x)
        return x