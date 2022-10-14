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
        total_pixels = x.shape[0] * x.shape[-2] * x.shape[-1]
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
            y1 = distance_transform_edt(x[:, i].cpu().detach().numpy())
            d1[:, i] = torch.tensor(y1, **factory_kwargs)

            y2 = distance_transform_edt(x[:, i].cpu().detach().numpy(), sampling=2)
            d2[:, i] = torch.tensor(y2, **factory_kwargs)

        d = 1.0 * self.w0 * torch.exp(-0.5 * ((d1 + d2) / self.sigma) ** 2)
        return d


# https://github.com/hayashimasa/UNet-PyTorch


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from mri_pituitary.data_loader import MRIPituitaryDataset
    from matplotlib.colors import ListedColormap
    from skimage.util import montage
    import matplotlib.pyplot as plt
    import os
    patient_num = 2
    os.chdir('/Users/elizabethnewman/OneDrive - Emory University/MRI - Labelling/With boxes/' + str(patient_num))

    img_dir = '.'
    my_data = MRIPituitaryDataset(img_dir)

    _, my_masks = my_data[:9]
    my_masks = torch.tensor(my_masks)
    my_masks = my_masks.permute(0, 3, 1, 2)

    cmap = ListedColormap(['red', 'green', 'blue', 'black'])
    img = montage(my_masks.argmax(dim=1).squeeze(), grid_shape=(3, 3))
    plt.imshow(img, cmap=cmap)
    plt.show()

    my_map = WeightMap()
    w = my_map(my_masks)

    img = montage(w[:, 0], grid_shape=(3, 3))
    plt.imshow(img)
    plt.show()





