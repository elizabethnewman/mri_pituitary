import torch
import math
from torch import Tensor
from torch.nn import Module
from typing import Tuple, List, Union


def setup_data_splits(images, masks, n=80, p=0.8, seed=42, dtype=None, device=None):
    """
    n : number of training and validation images
    p : percentage of training and testing images
    seed : for reproducibility

    """
    factory_kwargs = {'dtype': dtype, 'device': device}
    # prepare data
    if seed is not None:
        torch.manual_seed(seed)

    train_patients_images, train_patients_masks = images[:n], masks[:n]
    test_patients_images, test_patients_masks = images[n:], masks[n:]

    # split training images into training and validation
    n_train = int(math.ceil(p * train_patients_images.shape[0]))

    # shuffle
    idx = torch.randperm(train_patients_images.shape[0])
    train_patients_images = train_patients_images[idx]
    train_patients_masks = train_patients_masks[idx]

    # normalize images here
    images_train = torch.tensor(train_patients_images[:n_train].transpose(0, 3, 1, 2)).to(*factory_kwargs)
    masks_train = torch.tensor(train_patients_masks[:n_train].transpose(0, 3, 1, 2)).to(*factory_kwargs)

    images_val = torch.tensor(train_patients_images[n_train:].transpose(0, 3, 1, 2)).to(*factory_kwargs)
    masks_val = torch.tensor(train_patients_masks[n_train:].transpose(0, 3, 1, 2)).to(*factory_kwargs)

    images_test = torch.tensor(test_patients_images.transpose(0, 3, 1, 2)).to(*factory_kwargs)
    masks_test = torch.tensor(test_patients_masks.transpose(0, 3, 1, 2)).to(*factory_kwargs)

    return images_train, masks_train, images_val, masks_val, images_test, masks_test


def seed_everything(seed):
    # option to add numpy, random, etc.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_getattr(obj: Module, names: Tuple or List or str):
    r"""
    Get specific attribute of module at any level
    """
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return getattr(obj, names)
    else:
        return module_getattr(getattr(obj, names[0]), names[1:])


def module_setattr(obj: Module, names: Tuple or List, val: Union[Tensor, None]):
    r"""
    Set specific attribute of module at any level
    """
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return setattr(obj, names, val)
    else:
        return module_setattr(getattr(obj, names[0]), names[1:], val)


def extract_data(net: Module, attr: str = 'data') -> (torch.Tensor, Tuple, Tuple):
    """
    Extract data stored in specific attribute and store as 1D array
    """
    theta = ()
    for name, w in net.named_parameters():
        if getattr(w, attr) is not None:
            w = getattr(w, attr)

        theta += (w.reshape(-1),)

    return torch.cat(theta)


def insert_data(net: Module, theta: torch.Tensor, attr: str = 'data') -> None:
    """
    Insert 1D array of data into specific attribute
    """
    count = 0
    for name, w in net.named_parameters():
        name_split = name.split('.')
        n = w.numel()
        module_setattr(net, name_split + [attr], theta[count:count + n].reshape(w.shape))
        count += n


def none_grad(net: Module) -> None:
    """
    Insert 1D array of data into specific attribute
    """
    count = 0
    for name, w in net.named_parameters():
        name_split = name.split('.')
        n = w.numel()
        module_setattr(net, name_split + ['grad'], None)
        count += n


def get_num_parameters(net: Module) -> None:
    """
    Insert 1D array of data into specific attribute
    """
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count
