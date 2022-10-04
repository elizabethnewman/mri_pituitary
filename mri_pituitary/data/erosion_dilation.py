import numpy as np
import torch
import torch.nn.functional as F


def dilation(image, bg, kernel_size=3, pad=1):
    # MAIN IDEA: apply averaging filter to image and ensure border is considered backgroun
    # image is a background image here

    # get device and dtype
    factory_kwargs = {'device': image.device, 'dtype': image.dtype}

    K = torch.ones((1, 1, kernel_size, kernel_size), **factory_kwargs)
    K /= torch.sum(K)
    eps = 0.001 * torch.max(K).item()

    image = F.conv2d(image, K, padding=(pad, pad))
    image = 1.0 * ((image + bg) > eps)

    return image


def erosion(image, im, kernel_size=3, pad=1):
    # MAIN IDEA: apply averaging filter to image and ensure border is considered backgroun
    # image is a background image here

    # get device and dtype
    factory_kwargs = {'device': image.device, 'dtype': image.dtype}

    K = torch.ones((1, 1, kernel_size, kernel_size), **factory_kwargs)
    K /= torch.sum(K)
    eps = 0.001 * torch.max(K).item()

    image = F.conv2d(image, K, padding=(pad, pad))
    image = 1.0 * ((image + im) > 1 - eps)

    return image


def getBackground(image, bgimage=None, imin=-400, imax=400,
                  maxIter=200, kernel_size=3, pad=10, verbose=False):
    # MAIN IDEA: create a background mask (1's and 0's) and apply it to image
    #   (1) create a background image with all zeros and a border of ones (called bg and initializes bgImage)
    #       (1a) additionally, zero out entries that are extreme values in the original image
    #   (2) apply averaging convolution to bgImage that will move the border inward
    #       (2a) mask extreme values again
    #       (2b) make sure no values are too small
    #       (2c) repeat to bring into center of image
    #   (3) apply morphological transformations to refine mask

    # get device and dtype
    factory_kwargs = {'device': image.device, 'dtype': image.dtype}

    # 1 - mask to block out extreme values
    mask = 1.0 * ((image > imin) * (image < imax))

    # put ones around edges
    if bgimage is None:
        # create image of zeros with border of ones of thickness=pad
        bgimage = torch.ones_like(image)
        bgimage[:, :, pad:-pad, pad:-pad] = 0.0

        # mask extreme values
        bgimage *= 1.0 * (1 - mask)

    # initial background image
    bg = 1.0 * bgimage

    # create averaging convolutional filter
    K = torch.ones((1, 1, kernel_size, kernel_size), **factory_kwargs)
    K /= torch.sum(K)
    eps = 0.001 * torch.max(K).item()

    # GOAL: make a background mask
    flag1 = -1
    for k in range(maxIter):
        bgOld = 1.0 * bgimage

        # apply averaging filter to background image
        bgimage = F.conv2d(bgimage, K, padding=(pad, pad))

        # mask extreme values
        bgimage *= 1.0 * (1 - mask)

        # turn background image into 1's for values greater than eps and 0's otherwise
        bgimage = 1.0 * (bgimage > eps)

        # get the error between current background image and
        dbg = torch.sum(torch.abs(bgimage - bgOld)).item()
        if verbose:
            print("k=%d, dbg: %d" % (k, dbg))
        if dbg == 0:
            flag1 = 0
            break

    # GOAL:
    flag2 = -1
    for k in range(10):
        bgOld = 1.0 * bgimage

        # grow the mask
        bgimage = dilation(bgimage, bg, kernel_size=kernel_size, pad=pad)

        # erode
        bgimage = erosion(bgimage, bg, kernel_size=kernel_size, pad=pad)

        dbg = torch.sum(torch.abs(bgimage - bgOld)).item()
        if verbose:
            print("k=%d, dbg: %d" % (k, dbg))
        if dbg == 0:
            flag2 = 0
            break

    return bgimage, flag1, flag2


def getPituitary(image, bgimage, threshold=0, kernel_size=3, verbose=False, pad=1):
    # get device and dtype
    # factory_kwargs = {'device': image.device, 'dtype': image.dtype}

    image[bgimage == 1] = 0

    image[image > threshold] = 0
    image[image < threshold] = 1

    im = 1.0 * image
    flag = -1
    for k in range(10):
        imOld = 1.0 * image

        # grow the mask
        image = dilation(image, im, kernel_size=kernel_size, pad=pad)

        # erode
        image = erosion(image, im, kernel_size=kernel_size, pad=pad)

        dim = torch.sum(torch.abs(image - imOld)).item()
        if verbose:
            print("k=%d, dbg: %d" % (k, dim))
        if dim == 0:
            flag = 0
            break

    # close holes
    image = dilation(image, im, kernel_size=13, pad=6)
    image = erosion(image, im, kernel_size=13, pad=6)

    return image, flag


def convert2grayscale(images, coeff=(0.2989, 0.5870, 0.1140)):
    if images.shape[-1] == 3:
        images = (coeff[0] * images[..., 0]
                  + coeff[1] * images[..., 1]
                  + coeff[2] * images[..., 2])
        return images[..., np.newaxis]
    else:
        return images


if __name__ == "__main__":
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision.transforms import CenterCrop

    patient_num = 4

    os.chdir('/Users/elizabethnewman/OneDrive - Emory University/MRI - Labelling/With boxes/' + str(patient_num))

    N, C, H, W = 40, 3, 768, 1024

    img_num = '%.3d' % 10

    filename = str(patient_num) + '.' + str(img_num) + '.jpeg'
    p_img = np.array(Image.open(filename))
    img = torch.tensor(p_img)
    img = convert2grayscale(img.unsqueeze(0))
    print(img.shape)
    img = torch.tensor(img).permute(0, 3, 1, 2)
    img = img[:, :, 300:-50, 300:-300]
    # img = CenterCrop([600, 600])(img)

    print(img.shape)

    plt.figure()
    plt.imshow(img.squeeze())
    plt.show()
    #
    bg = getBackground(img)[0]

    plt.figure()
    plt.imshow(bg.squeeze())
    plt.show()

    pit = getPituitary(img, bg)[0]

    plt.figure()
    plt.imshow(pit.squeeze())
    plt.show()



