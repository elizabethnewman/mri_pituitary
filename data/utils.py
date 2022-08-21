import numpy as np
from PIL import Image
import math
import cv2 as cv


def make_masks(label_imgs, threshold=0.5):
    # create probabilities
    masks = label_imgs / np.sum(label_imgs, axis=-1, keepdims=True)

    # only store maximum value
    masks = masks * (masks == masks.max(axis=-1)[..., np.newaxis])

    # threshold (find 3 main classes)
    masks = (masks > threshold)

    # find background
    background = (np.sum(masks, axis=-1, keepdims=True) < 1)

    masks = np.concatenate((masks, background), axis=-1)

    return 1 * masks


def rgb2gray(filename, cutoff=0):
    img = np.array(Image.open(filename).convert('L'))

    if cutoff > 0:
        img = img[cutoff:-cutoff, cutoff:-cutoff]
    return img


def get_box(patient_box, min_size=250):
    # TODO: check sizes

    # get top and bottom boundaries
    top = np.argmin(patient_box, axis=1)
    top = np.apply_along_axis(lambda x: np.min(x[x > 0]), 1, top)

    bottom = np.argmin(np.flip(patient_box, axis=1), axis=1)
    bottom = np.apply_along_axis(lambda x: np.min(x[x > 0]), 1, bottom)
    bottom = patient_box.shape[1] - bottom

    # make sure sizes are the same
    d = np.abs(bottom - top)
    pad = np.ceil(min_size - d) // 2
    top = top - pad
    # bottom = bottom + (min_size - d + pad)
    bottom = top + min_size

    # get left and right boundaries
    left = np.argmin(patient_box, axis=2)
    left = np.apply_along_axis(lambda x: np.min(x[x > 0]), 1, left)

    right = np.argmin(np.flip(patient_box, axis=2), axis=2)
    right = np.apply_along_axis(lambda x: np.min(x[x > 0]), 1, right)
    right = patient_box.shape[2] - right

    d = np.abs(right - left)
    pad = np.ceil(min_size - d) // 2
    left = left - pad
    right = left + min_size
    # right += pad

    # concatenate
    boundaries = np.hstack((left.reshape(-1, 1), right.reshape(-1, 1), top.reshape(-1, 1), bottom.reshape(-1, 1)))
    boundaries = boundaries.astype(int)
    return boundaries


def crop_img(img, boundaries):

    H = abs(boundaries[0, 3] - boundaries[0, 2])
    W = abs(boundaries[0, 1] - boundaries[0, 0])

    cropped_imgs = np.zeros([img.shape[0], H, W, img.shape[-1]])
    for i in range(img.shape[0]):
        cropped_imgs[i] = img[i][boundaries[i, 2]:boundaries[i, 3], boundaries[i, 0]:boundaries[i, 1]]

    return cropped_imgs


def normalize_image(img, nrm_type):
    if nrm_type == 'image_standardization':
        # https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        if img.ndim == 4:
            img = img - img.mean(axis=(1, 2, 3), keepdims=True)
            img = img / np.maximum(img.std(axis=(1, 2, 3), keepdims=True), 1 / math.sqrt(img[0].size))
        else:
            img = img - img.mean()
            img = img / max(img.std(), 1 / math.sqrt(img.size))

    elif nrm_type == 'clahe':
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    else:
        # rescale
        img = (img - img.min()) / (img.max() - img.min())

    return img


def convert_raw2ML(img, mask, boundaries, names, cm=None, nrm_type='image_standardization'):

    # if cm is None:
    #     cm = (img.shape[1:3] // 2) * np.ones(img.shape[0], 2)
    #
    # cm = np.floor(cm).astype(int)

    # find lower and upper bounds
    # lower = np.maximum(cm - np.array([box[0], box[2]]).reshape(1, -1), 0)
    # upper = np.maximum(cm + np.array([box[1], box[3]]).reshape(1, -1), 0)
    #
    # # compute difference (have to add a catch if reaching boundary)
    # d = np.min(upper - lower, axis=0)

    # crop images based on center of mass
    img2 = crop_img(img, boundaries)
    mask2 = crop_img(mask, boundaries)

    # img2 = np.zeros((img.shape[0], box[0] + box[2], box[1] + box[3], img.shape[-1])).astype(img.dtype)
    # mask2 = np.zeros((mask.shape[0], box[0] + box[2], box[1] + box[3], mask.shape[-1])).astype(img.dtype)
    # for i in range(img.shape[0]):
    #     img2[i] = img[i, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]]
    #     mask2[i] = mask[i, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]]

    info = {'data': normalize_image(img2, nrm_type).astype(np.float32),
            'mask': mask2.astype(int),
            'orig_shape': img.shape,
            'boundaries': boundaries,
            'center_of_mass': None,
            'id': names}

    return info


def create_full_mask(info):

    mask_full = np.zeros(info['orig_shape'][0:3] + (4,), dtype=info['mask'].dtype)
    mask_full[..., -1] = 1
    mask_full[:, info['cutoffs'][0]:info['cutoffs'][1], info['cutoffs'][2]:info['cutoffs'][3]] = info['mask']

    return mask_full
