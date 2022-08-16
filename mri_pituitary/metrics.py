import torch


def get_dice(X: torch.Tensor, Y: torch.Tensor, k: int):
    num = 2 * ((X == k) * (Y == k)).sum()
    den = (X == k).sum() + (Y == k).sum()
    return num / max(den, 1)


def get_labels(X: torch.Tensor):
    """
    Convert output probabilities to label per pixel
    :param X: output of network of size N x C x H x W
    :type X: torch.Tensor
    :return: label per pixel of size N x H x W
    :rtype: torch.Tensor
    """
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
