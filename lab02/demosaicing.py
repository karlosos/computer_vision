import matplotlib.pyplot as plt
import numpy as np
from funcy import print_durations

from cfa import img_to_cfa
from cfa import bayer_channel_for_index


@print_durations
def demosaicing(mosaic, interpolation_method):
    """
    Demosaicing image
    :param mosaic: one channel image created with Bayer filter
    :param interpolation_method: function used for interpolating pixels in each channel
    :return: reconstructed image
    """
    height, width = mosaic.shape

    bayer_mosaics = np.full((height, width, 3), np.nan)

    for i in range(height):
        for j in range(width):
            channel = bayer_channel_for_index(i, j)
            bayer_mosaics[i, j, channel] = mosaic[i, j]

    output = interpolation_method(bayer_mosaics)

    # Plot interpolated channels
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(output[:, :, 0])
    axes[1].imshow(output[:, :, 1])
    axes[2].imshow(output[:, :, 2])
    plt.show()

    return output


@print_durations()
def simple_interpolation(bayer_mosaics):
    """
    Simple interpolation using a mean of 9 nearest neighbors
    :param bayer_mosaics: input image of shape (height, width, 3) with np.nan where value is unknown
    :return: rgb image with interpolated values in channels
    """
    height, width, _ = bayer_mosaics.shape
    output = np.copy(bayer_mosaics)
    for i in range(height):
        for j in range(width):
            # Interpolate red
            if bayer_channel_for_index(i, j) != 0:
                output[i, j, 0] = mean_neighbors(bayer_mosaics[:, :, 0], i, j)
            # Interpolate green
            if bayer_channel_for_index(i, j) != 1:
                output[i, j, 1] = mean_neighbors(bayer_mosaics[:, :, 1], i, j)
            # Interpolate blue
            if bayer_channel_for_index(i, j) != 2:
                output[i, j, 2] = mean_neighbors(bayer_mosaics[:, :, 2], i, j)
    return output.astype(int)


def mean_neighbors(channel, i, j):
    """
    Calculate mean from 9 nearest neighbors for a channel

    :param channel: matrix with (height, width) shape - single channel of mosaic. Filled with values and np.nan
    :param i: horizontal index (height)
    :param j: vertical index (width)
    :return: mean of 9 nearest neighbors
    """
    height, width = channel.shape
    lower_bound = i - 1 if i - 1 >= 0 else 0
    upper_bound = i + 1 if i + 1 < height else height - 1
    left_bound = j - 1 if j - 1 >= 0 else 0
    right_bound = j + 1 if j + 1 < width else width - 1
    return np.nanmean(channel[lower_bound:upper_bound+1, left_bound:right_bound+1])


def main():
    img = plt.imread("./data/image1.jpg")
    plt.imshow(img)
    plt.show()

    img_bayer = img_to_cfa(img)
    plt.imshow(img_bayer)
    plt.show()

    output = demosaicing(img_bayer, interpolation_method=simple_interpolation)
    plt.imshow(output)
    plt.show()


if __name__ == "__main__":
    main()
