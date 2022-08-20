import torch


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
    if masks.ndim < 2:
        num_classes = torch.unique(masks, eturn_counts=True)[-1]
    else:
        num_classes = masks.shape[1]

    dice_values = []
    for k in range(num_classes):
        dice_values.append(get_dice(predicted_labels, true_labels, k))

    return Jc.item(), acc.item(), dice_values


def get_dice(X: torch.Tensor, Y: torch.Tensor, k: int):
    num = 2 * ((X == k) * (Y == k)).sum()
    den = (X == k).sum() + (Y == k).sum()
    return num.item() / max(den.item(), 1)


def get_labels(X: torch.Tensor):
    """
    Convert output probabilities to label per pixel
    :param X: output of network of size N x C x H x W
    :type X: torch.Tensor
    :return: label per pixel of size N x H x W
    :rtype: torch.Tensor
    """
    if X.ndim < 2:
        # assume we have class labels, not one-hot vectors
        return X
    else:
        return torch.argmax(X, dim=1)


def get_accuracy(X: torch.Tensor, Y: torch.Tensor):
    """
    Compute the accuracy of network labeling
    :param X: output of the network
    :type X: torch.Tensor
    :param Y:
    :type Y:
    :return:
    :rtype:
    """
    predicted_labels = get_labels(X)
    true_labels = get_labels(Y)
    acc = 100 * (predicted_labels == true_labels).sum() / predicted_labels.numel()
    return acc


def confusion_matrix(X: torch.Tensor, Y: torch.Tensor):
    predicted_labels = get_labels(X).reshape(-1)
    true_labels = get_labels(Y).reshape(-1)
    # print(pred_labels.shape)

    n_labels = len(torch.unique(true_labels))
    # print(n_labels)
    conf_mat = torch.zeros(n_labels, n_labels)

    # for i in range(pred_labels.numel()):
    #   j1 = pred_labels[i]
    #   j2 = true_labels[i]
    #   conf_mat[j1, j2] += 1

    for i in range(n_labels):
        for j in range(n_labels):
            conf_mat[i, j] = torch.sum((predicted_labels == i) * (true_labels == j))

    return conf_mat
