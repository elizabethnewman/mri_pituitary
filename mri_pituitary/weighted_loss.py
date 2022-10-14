import torch
import torch.nn.functional as F
from mri_pituitary.weight_map_class import WeightMap


class WeightedLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.map = WeightMap()

    def forward(self, x, y):
        w = self.map(y)
        x = y * F.log_softmax(x, dim=1)
        x = w * x
        # TODO: - or +
        return -x.mean(dim=(0, 2, 3)).sum()
