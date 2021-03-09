"""
Function for creating CFA image based on Bayer Filter
More about Bayer filter here: https://www.wikiwand.com/en/Bayer_filter
"""

import numpy as np

RED = 0
GREEN = 1
BLUE = 2


def img_to_cfa(img):
    """
    Creates mosaic from rgb image

    Simulates output of a sensor with a Bayer filter

    :param img: input image with 3 channels with shape (height, width, 3)
    :return: mosaic image with shape (height, width)
    """
    mosaic = np.zeros((img.shape[0], img.shape[1]))
    height, width, _ = img.shape

    for i in range(height):
        for j in range(width):
            channel = bayer_channel_for_index(i, j)
            mosaic[i, j] = img[i, j, channel]

    return mosaic


def bayer_channel_for_index(i, j):
    """
    Calculates channel based on Bayer arrangement of color pixels

    :param i: position in vertical axis (height)
    :param j: position in horizontal axis (width)
    :return: channel number, RED = 0, GREEN = 1, BLUE = 2
    """
    if i % 2 == 0:
        if j % 2 == 0:
            return GREEN
        else:
            return RED
    else:
        if j % 2 != 0:
            return GREEN
        else:
            return BLUE
