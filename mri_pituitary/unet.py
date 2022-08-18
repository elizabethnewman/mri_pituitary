import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2=None, kernel_size=(3, 3), padding=(0, 0), **kwargs):
        super(Conv2dReLU, self).__init__()

        if out_channels2 is None:
            out_channels2 = out_channels1

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size, padding=padding, **kwargs)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size, padding=padding, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):

    def __init__(self, channels, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super(Encoder, self).__init__()

        self.channels = channels
        self.blocks = nn.ModuleList([
            Conv2dReLU(channels[i], channels[i + 1], **factory_kwargs) for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.blocks:
            # apply convolution
            x = block(x)

            # store featurs
            features.append(x.clone().detach().requires_grad_(False))

            # pool layer
            x = self.pool(x)

        return x, features


class Decoder(nn.Module):

    def __init__(self, channels, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super(Decoder, self).__init__()

        self.channels = channels
        self.blocks = nn.ModuleList(
            [Conv2dReLU(channels[i], channels[i + 1], **factory_kwargs)
             for i in range(len(channels) - 1)])
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=(2, 2), stride=(2, 2), **factory_kwargs)
            for i in range(len(channels) - 1)])

    def forward(self, x, features):
        for i in range(len(self.channels) - 1):
            # up-conv
            x = self.upconvs[i](x)

            # crop
            _, _, H, W = x.shape
            feat = CenterCrop([H, W])(features[i])

            # concatenate
            x = torch.cat((x, feat), dim=1)

            # apply convolution
            x = self.blocks[i](x)

        return x


class UNet(nn.Module):

    def __init__(self,
                 enc_channels=(3, 64, 128, 256, 512),
                 dec_channels=(512, 256, 128, 64),
                 intrinsic_channels=1024,
                 num_classes=3,
                 resize=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(UNet, self).__init__()
        self.encoder = Encoder(enc_channels, **factory_kwargs)
        self.decoder = Decoder((intrinsic_channels,) + dec_channels, **factory_kwargs)

        # intrinsic layers
        self.conv = Conv2dReLU(enc_channels[-1], intrinsic_channels, **factory_kwargs)
        self.output_map = nn.Conv2d(dec_channels[-1], num_classes, kernel_size=(1, 1), **factory_kwargs)

        # final resize
        self.resize = resize

    def forward(self, x):
        # starting size
        _, _, H, W = x.shape

        # encoder
        x, features = self.encoder(x)

        # intrinsic layers
        x = self.conv(x)

        # decoder
        x = self.decoder(x, features[::-1])

        # final map
        x = self.output_map(x)

        # resize option
        if self.resize:
            x = F.interpolate(x, (H, W))

        return x

    