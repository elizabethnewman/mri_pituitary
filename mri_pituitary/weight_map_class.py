import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt


class WeightMap(nn.Module):

    def __init__(self, w0=10, sigma=5):
        super(WeightMap, self).__init__()
        self.w0 = w0
        self.sigma = sigma

    def forward(self, x):
        wc = self.calculate_weights(x)
        d = self.distance(x)
        return wc + d

    def calculate_weights(self, x):
        num_pixels_per_class = x.sum(dim=(0, 2, 3), keepdim=True)
        total_pixels = x.numel()
        w = num_pixels_per_class / total_pixels
        return w * torch.ones_like(x)

    def distance(self, x: torch.Tensor):
        # assume x is a mask of size (num_samples, num_classes, H, W)
        # measures the distance to the border of the nearest cell

        factory_kwargs = {'device': x.device, 'dtype': x.dtype}

        # create binary image
        d1 = torch.zeros_like(x)
        d2 = torch.zeros_like(x)
        for i in range(x.shape[1]):
            # 1's represent inside class, 0's represent outside class
            y1 = distance_transform_edt(x[:, :, i].detach().numpy())
            d1[:, :, i] = torch.tensor(y1, **factory_kwargs)

            y2 = distance_transform_edt(x[:, :, i].detach().numpy(), sampling=2)
            d2[:, :, i] = torch.tensor(y2, **factory_kwargs)

        d = self.w0 * torch.exp(-0.5 * ((d1 + d2) / self.sigma) ** 2)
        return d



# https://github.com/hayashimasa/UNet-PyTorch
