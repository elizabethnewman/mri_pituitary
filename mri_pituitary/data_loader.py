import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.transforms.functional import crop
from torchvision.io import read_image
from mri_pituitary.data.utils import rgb2gray, make_masks, crop_img, convert_raw2ML, get_box, normalize_image
import numpy as np
import glob


class MRIPituitaryDataset(Dataset):
    def __init__(self, img_dir, gray=True, nrm_type=None, transform=None, target_transform=None):
        self.img_dir = img_dir
        tmp = glob.glob(img_dir + '/*.jpeg', recursive=False)
        tmp.sort()
        self.img_names = tmp
        self.nrm_type = nrm_type
        self.gray = gray
        self.transform = transform

        self.target_transform = target_transform
        if transform is not None and target_transform is None:
            self.target_transform = transform

    def __len__(self):
        return len(self.img_names) // 3

    def __getitem__(self, idx):

        images = ()
        masks = ()
        boxes = ()

        if isinstance(idx, int):
            a = idx
            b = idx + 1
        else:
            a = idx.start if idx.start is not None else 0
            b = idx.stop if idx.stop is not None else self.__len__()

        for i in range(a, b):
            # img_path = os.path.join(self.img_dir, self.img_names[3 * i])
            img_path = self.img_names[3 * i]

            if self.gray:
                image = rgb2gray(img_path)[..., np.newaxis]
            else:
                image = read_image(img_path)

            # mask_path = os.path.join(self.img_dir, self.img_names[3 * i + 1])
            mask_path = self.img_names[3 * i + 1]
            mask = read_image(mask_path).permute((1, 2, 0))

            # box_path = os.path.join(self.img_dir, self.img_names[3 * i + 2])
            box_path = self.img_names[3 * i + 2]

            if self.gray:
                box = rgb2gray(box_path)
            else:
                box = read_image(box_path)

            images += (image[np.newaxis, ...],)
            masks += (mask[np.newaxis, ...],)
            boxes += (box[np.newaxis, ...],)

        # concatenate
        images = np.concatenate(images, axis=0)
        masks = make_masks(np.concatenate(masks, axis=0))
        boxes = get_box(np.concatenate(boxes, axis=0))

        # crop
        images = crop_img(images, boxes)
        masks = crop_img(masks, boxes)

        # normalize
        images = normalize_image(images, self.nrm_type).astype(np.float32)
        masks = masks.astype(int)

        # permute to PyTorch orientation
        # images = images.transpose(0, 3, 1, 2)
        # masks = masks.transpose(0, 3, 1, 2)

        if self.transform:
            images2 = ()
            for i in range(images.shape[0]):
                images2 += (self.transform(images[i]),)

            images = torch.cat(images2, dim=0)

        if self.target_transform:
            masks2 = ()
            for i in range(images.shape[0]):
                masks2 += (self.target_transform(masks[i]),)

            masks = torch.cat(masks2).float()

        sample = (images, masks)
        return sample


if __name__ == "__main__":
    from torchvision import datasets, transforms
    patient_num = 2
    os.chdir('/Users/elizabethnewman/OneDrive - Emory University/MRI - Labelling/With boxes/' + str(patient_num))

    img_dir = '.'
    my_data = MRIPituitaryDataset(img_dir)

    print(my_data.__len__())

    sample = my_data[0:3]


    # check with transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    target_transform = transforms.ToTensor()

    my_data2 = MRIPituitaryDataset(img_dir, transform=transform, target_transform=target_transform)
    sample2 = my_data2[:3]

    # check data loader
    train_loader = torch.utils.data.DataLoader(my_data2, batch_size=5)

