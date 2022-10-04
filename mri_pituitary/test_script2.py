import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

import os
import pickle

from mri_pituitary.unet import UNet
from mri_pituitary.utils import seed_everything, extract_data, get_num_parameters
from mri_pituitary.metrics import get_dice, get_accuracy, get_labels, compute_metrics
from mri_pituitary.data.visualization import plot_mask, plot_output_features
from mri_pituitary.lbfgs import LBFGS
from mri_pituitary.objective_function import ObjectiveFunction

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%% upload data


os.chdir('/Users/elizabethnewman/Desktop/DesktopClean-August26/tmp/')

info2 = pickle.load(open("patient-2.p", "rb" ))

info4 = pickle.load(open("patient-4.p", "rb" ))

info30 = pickle.load(open("patient-30.p", "rb" ))

images = np.concatenate((info4['data'], info2['data'], info30['data']), axis=0)
masks = np.concatenate((info4['mask'], info2['mask'], info30['mask']), axis=0)

#%%

seed_everything(42)

shuffle = True

n = 80

if shuffle:
    idx = torch.randperm(images.shape[0])
else:
    idx = torch.arange(images.shape[0])

# split into training and testing
train_patients_images, train_patients_masks = images[idx[:n]], masks[idx[:n]]
test_patients_images, test_patients_masks = masks[idx[n:]], masks[idx[n:]]

# split into training and validation
n_train = int(np.round(0.8 * train_patients_images.shape[0]))
print(n_train)

# normalize images here
images_train = torch.tensor(train_patients_images[:n_train].transpose(0, 3, 1, 2)).to(torch.float32).to(device)
masks_train = torch.tensor(train_patients_masks[:n_train].transpose(0, 3, 1, 2)).to(torch.float32).to(device)

images_val = torch.tensor(train_patients_images[n_train:].transpose(0, 3, 1, 2)).to(torch.float32).to(device)
masks_val = torch.tensor(train_patients_masks[n_train:].transpose(0, 3, 1, 2)).to(torch.float32).to(device)

#%% create loss

seed_everything(42)

total_pixels = images_train.shape[2] * images_train.shape[3]
batch_size = images_train.shape[0]
pixels_per_class = torch.sum(masks_train, dim=(0, 2, 3))
weight = total_pixels / pixels_per_class
# weight = torch.tensor([1.0, 1.0, 1.0, 1.0])

print(weight)
loss = nn.CrossEntropyLoss(weight=weight.to(device), reduction='mean')

print(loss)

#%% evaluation


def evaluate(net, loss, images_train, masks_train,images_val, masks_val, iter, g0nrm=1.0):
    # p = extract_data(net, 'data')
    # pnrm = torch.norm(p).item()
    # g = extract_data(net, 'grad')
    # gnrm = torch.norm(g).item()

    net.eval()
    with torch.no_grad():
        Jc_train, acc_train, dice_train = compute_metrics(images_train, masks_train, net, loss)
        Jc_val, acc_val, dice_val = compute_metrics(images_val, masks_val, net, loss)
        values = [iter, 0, 0, 1.0 / 1.0,
                  Jc_train, acc_train,
                  dice_train[0], dice_train[1], dice_train[2], dice_train[3], sum(dice_train) / len(dice_train),
                  Jc_val, acc_val, dice_val[0], dice_val[1], dice_val[2], dice_val[3], sum(dice_val) / len(dice_val)]

    return values

#%%

seed_everything(20)


n_classes = masks.shape[-1]
w = 8
net = UNet(enc_channels=(1, w, 2 * w, 4 * w), dec_channels=(4 * w, 2 * w, w), intrinsic_channels=8 * w, num_classes=4).to(device)
# print(net)

alpha = 0.0
f = ObjectiveFunction(net, loss, alpha=alpha)
f.mean_type = 'total_mean'

n = get_num_parameters(net)
optimizer = LBFGS(n, device=device)
optimizer.ls.alpha_max = 5.0
print(optimizer)

seed_everything(42)
p = extract_data(net, 'data')
p_opt, info = optimizer.solve(f, p, images_train, masks_train, show_mask_plots=True)
