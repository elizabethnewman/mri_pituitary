import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_mask(mask, axis=0, show_colorbar=True):
    # assume mask is of size C x H x W
    # https://riptutorial.com/matplotlib/example/20692/custom-discrete-colormap

    if mask.shape[0] == 4:
        cmap = ListedColormap(['red', 'green', 'blue', 'black'])
        vmax = 3
    else:
        cmap = ListedColormap(['white', 'black'])
        vmax = 1

    mm = mask.argmax(axis=axis)
    plt.imshow(mm, cmap=cmap, vmin=0, vmax=vmax)
    plt.axis('off')

    if show_colorbar:
        cbar = plt.colorbar()
        cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar.set_ticklabels(('tumor?', 'stalk?', 'gland?', 'none'))


def plot_output_features(out):
    plt.figure()

    for i, name in enumerate(['red', 'green', 'blue', 'background']):
        plt.subplot(2, 2, i + 1)
        plt.imshow(out[i].detach().cpu())
        plt.axis('off')
        plt.colorbar()
        plt.title(name)
