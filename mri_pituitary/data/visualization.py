import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_mask(mask, axis=0, show_colorbar=True):
    # assume mask is of size C x H x W
    # https://riptutorial.com/matplotlib/example/20692/custom-discrete-colormap
    cmap = ListedColormap(['red', 'green', 'blue', 'black'])
    mm = mask.argmax(axis=axis)
    plt.imshow(mm, cmap=cmap, aspect='auto', vmin=0, vmax=3)
    plt.axis('off')

    if show_colorbar:
        cbar = plt.colorbar()
        cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar.set_ticklabels(('tumor?', 'stalk?', 'gland?', 'none'))

