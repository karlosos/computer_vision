import matplotlib.pyplot as plt
import numpy as np
from funcy import print_durations
from numba import jit

from cfa import img_to_cfa
from cfa import bayer_channel_for_index
from convolution import add_padding
from metrics import psnr

# Supressing deprecation warnings from numba
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaWarning,
)
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


@print_durations
def demosaicing_simple(mosaic, interpolation_method):
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

    return output


@jit
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


@jit
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
    return np.nanmean(
        channel[lower_bound : upper_bound + 1, left_bound : right_bound + 1]
    )


@print_durations()
@jit
def demosaicing_malvar(mosaic):
    """
    Malvar algorithm

    Henrique S. Malvar, Li-wei He, and Ross Cutler , High-Quality Linear Interpolation For Demosaicing Of Bayer-Patterned Color Images
    :param bayer_mosaics: input image of shape (height, width, 3) with np.nan where value is unknown
    :return: rgb image with interpolated values in channels
    """
    mosaic = add_padding(mosaic, 2)
    height, width = mosaic.shape

    bayer_mosaics = np.full((height, width, 3), np.nan, dtype=int)

    GR_GB = (
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        / 8
    )

    Rg_RB_Bg_BR = (
        np.array(
            [
                [0, 0, 0.5, 0, 0],
                [0, -1, 0, -1, 0],
                [-1, 4, 5, 4, -1],
                [0, -1, 0, -1, 0],
                [0, 0, 0.5, 0, 0],
            ]
        )
        / 8
    )

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = (
        np.array(
            [
                [0, 0, -1.5, 0, 0],
                [0, 2, 0, 2, 0],
                [-1.5, 0, 6, 0, -1.5],
                [0, 2, 0, 2, 0],
                [0, 0, -1.5, 0, 0],
            ]
        )
        / 8
    )

    for i in range(2, height - 2):
        for j in range(2, width - 2):
            # Green channel
            if (
                bayer_channel_for_index(i, j) == 0 or bayer_channel_for_index(i, j) == 2
            ):  # if in Red or Blue
                c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                value = (GR_GB * c).sum()
                bayer_mosaics[i, j, 1] = int(value)
            else:
                bayer_mosaics[i, j, 1] = mosaic[i, j]
            # Red channel
            if bayer_channel_for_index(i, j) == 1:  # if R at green
                if i % 2 == 0 and j % 2 == 0:  # if R row and B column
                    c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                    value = (Rg_RB_Bg_BR * c).sum()
                    bayer_mosaics[i, j, 0] = int(value)
                else:
                    c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                    value = (Rg_BR_Bg_RB * c).sum()
                    bayer_mosaics[i, j, 0] = int(value)
            elif bayer_channel_for_index(i, j) == 2:  # if R at blue in B row B column
                c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                value = (Rb_BB_Br_RR * c).sum()
                bayer_mosaics[i, j, 0] = int(value)
            else:
                bayer_mosaics[i, j, 0] = mosaic[i, j]
            # Blue channel
            if bayer_channel_for_index(i, j) == 1:  # if B at green
                if i % 2 == 0 and j % 2 == 0:  # if R row and B column
                    c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                    value = (Rg_BR_Bg_RB * c).sum()
                    bayer_mosaics[i, j, 2] = int(value)
                else:
                    c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                    value = (Rg_RB_Bg_BR * c).sum()
                    bayer_mosaics[i, j, 2] = int(value)
            elif bayer_channel_for_index(i, j) == 0:  # if B at red in B row B column
                c = mosaic[i - 2 : i + 2 + 1, j - 2 : j + 2 + 1]
                value = (Rb_BB_Br_RR * c).sum()
                bayer_mosaics[i, j, 2] = int(value)
            else:
                bayer_mosaics[i, j, 2] = mosaic[i, j]

    return bayer_mosaics[2:-2, 2:-2, :]


def main():
    """
    Testing demosaicing on single image
    """
    img = plt.imread("./data/image2.jpg")
    plt.imshow(img)
    plt.title("Original image")
    plt.show()

    img_bayer = img_to_cfa(img)
    plt.imshow(img_bayer, cmap="gray")
    plt.title("CFA image")
    plt.show()

    output_simple = demosaicing_simple(
        img_bayer, interpolation_method=simple_interpolation
    )
    output_malvar = demosaicing_malvar(img_bayer)
    print(output_simple.shape)
    print(output_malvar.shape)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(output_simple)
    axes[0].set_title("Simple interpolation")
    axes[1].imshow(output_malvar)
    axes[1].set_title("Malvar algorithm")
    plt.show()

    psnr_simple = psnr(img, output_simple)
    psnr_malvar = psnr(img, output_malvar)

    print(f"Original to original PSNR: {psnr(img, img)}")
    print(f"Simple interpolation PSNR: {psnr_simple}")
    print(f"Malvar interpolation PSNR: {psnr_malvar}")


def compare_demosaicing():
    files = ["./data/image1.jpg", "./data/image2.jpg", "./data/image3.jpg"]

    data = {"image": [], "psnr_simple": [], "psnr_malvar": []}

    for f in files:
        img = plt.imread(f)
        img_bayer = img_to_cfa(img)

        output_simple = demosaicing_simple(
            img_bayer, interpolation_method=simple_interpolation
        )
        output_malvar = demosaicing_malvar(img_bayer)

        psnr_simple = psnr(img, output_simple)
        psnr_malvar = psnr(img, output_malvar)

        data["image"].append(f)
        data["psnr_simple"].append(psnr_simple)
        data["psnr_malvar"].append(psnr_malvar)

    import pandas as pd

    df = pd.DataFrame(data)
    df.to_csv("experiments_data.csv")
    print(df)
    print(df.mean())


if __name__ == "__main__":
    main()
    # compare_demosaicing()
