import numpy as np
import scipy.ndimage
from PIL import Image
from data.utils import rgb2gray, make_masks, normalize_image, convert_raw2ML, create_full_mask, get_box
import pickle
import os
import scipy.ndimage

# TODO: file names with and without spaces
patient_num = 2

os.chdir('/Users/elizabethnewman/OneDrive - Emory University/MRI - Labelling/With boxes/' + str(patient_num))

N, C, H, W = 40, 1, 768, 1024
n_classes = 4

patient_img = np.zeros((N, H, W, C))
patient_mask = np.zeros((N, H, W, 3))  # not including background
patient_box = np.zeros((N, H, W))  # not including background
center_mass = np.zeros((N, 2))
patient_files = []


count = 0
for i in range(1, 121, 3):
    # convert to grayscale

    # image
    img_num = '%.3d' % i
    filename = str(patient_num) + '.' + img_num + '.jpeg'

    if os.path.exists(filename):
        p_img = rgb2gray(filename)
        patient_files.append(filename)
        patient_img[count] = p_img[:, :, np.newaxis]
        # center_mass[count] = scipy.ndimage.center_of_mass(patient_img[count, :, :, 0])

    # mask
    img_num = '%.3d' % (i + 1)
    filename = str(patient_num) + '.' + img_num + '.jpeg'
    if os.path.exists(filename):
        p_mask = np.array(Image.open(filename))
        patient_mask[count] = p_mask

    # box
    img_num = '%.3d' % (i + 2)
    filename = str(patient_num) + '.' + img_num + '.jpeg'
    if os.path.exists(filename):
        p_box = rgb2gray(filename)
        patient_box[count] = p_box

    count += 1

patient_img = patient_img[:count].astype(np.float32)
patient_mask = make_masks(patient_mask[:count])
patient_boundaries = get_box(patient_box[:count])

print(os.getcwd())
# info = {'data': normalize_image(patient_img[:, 400:-150, 400:-400], 'image_standardization'),
#         'mask': patient_mask[:, 400:-150, 400:-400],
#         'orig_img_size': (patient_img.shape[1], patient_img.shape[2]),
#         'cutoffs': ((400, patient_img.shape[1] - 150), (400, patient_img.shape[2] - 400)),
#         'id': patient_files}
info = convert_raw2ML(patient_img, patient_mask, patient_boundaries, patient_files, cm=center_mass, nrm_type='image_standardization')

os.chdir('/Users/elizabethnewman/Desktop/tmp/')
print(os.getcwd())
pickle.dump(info, open("patient-" + str(patient_num) + ".p", "wb"))


#%%
import matplotlib.pyplot as plt

# cutoffs = (350, -150, 400, -400)
# info = convert_raw2ML(patient_img, patient_mask, patient_files,
#                       box=(350, -150, 400, -400), nrm_type='image_standardization')

n = patient_mask.shape[0]

plt.figure()
for i in range(min(40, n)):
    plt.subplot(5, 8, i + 1)
    plt.imshow(255 * patient_mask[i, :, :, :3])
    plt.axis('off')

plt.show()

plt.figure()
for i in range(min(40, n)):
    plt.subplot(5, 8, i + 1)
    plt.imshow(patient_img[i, :, :, 0])
    plt.axis('off')

plt.show()

plt.figure()
for i in range(min(40, n)):
    plt.subplot(5, 8, i + 1)
    plt.imshow(patient_box[i])
    plt.axis('off')

plt.show()

plt.figure()
for i in range(min(40, n)):
    plt.subplot(5, 8, i + 1)
    plt.imshow(255 * info['mask'][i, :, :, :3])
    plt.axis('off')

plt.show()

plt.figure()
for i in range(min(40, n)):
    plt.subplot(5, 8, i + 1)
    plt.imshow(info['data'][i, :, :, 0])
    plt.axis('off')

plt.show()
