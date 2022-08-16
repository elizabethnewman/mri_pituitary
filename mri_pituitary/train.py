import torch
from mri_pituitary.metrics import get_labels, get_dice, get_accuracy


def compute_metrics(images, masks, net, loss):
    # output features
    out = net(images)

    # loss
    Jc = loss(out, masks)

    # accuracy
    predicted_labels = get_labels(out)
    true_labels = get_labels(masks)
    acc = 100 * (predicted_labels == true_labels).sum() / predicted_labels.numel()

    # dice values
    dice_values = torch.zeros(masks.shape[1])
    for k in range(masks.shape[1]):
        dice_values[k] = get_dice(predicted_labels, true_labels, k)

    return Jc.item(), acc.item(), dice_values



