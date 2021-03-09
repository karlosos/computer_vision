import matplotlib.pyplot as plt
import numpy as np


"""
Generating CFA image
Mosaicing
"""
def img_to_cfa(img):
    mosaic = np.zeros((img.shape[0], img.shape[1]))
    height, width, _ = img.shape
    
    for i in range(height):
        for j in range(width):
            mosaic[i, j] = bayer_filter_pixel_value(img[i, j, :], i, j)

    return mosaic

def bayer_filter_pixel_value(rgb_tuple, i, j):
    RED = 0
    GREEN = 1
    BLUE = 2

    channel = bayer_channel_for_index(i, j)
    return rgb_tuple[channel]


def bayer_channel_for_index(i, j):
    RED = 0
    GREEN = 1
    BLUE = 2

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

"""
Demosaicing
"""
def demosaicing(mosaic):
    height, width = mosaic.shape

    # Separate mosaic into three martices 
    red = np.full_like(mosaic, np.nan)
    green = np.full_like(mosaic, np.nan)
    blue = np.full_like(mosaic, np.nan)
    bayer_mosaics = np.array([red, green, blue])

    for i in range(height):
        for j in range(width):
            channel = bayer_channel_for_index(i, j)
            bayer_mosaics[channel, i, j] = mosaic[i, j]

    rgb_channels = np.copy(bayer_mosaics)

    # Simple interpolation 
    for i in range(height):
        for j in range(width):
            # Interpolate red 
            if bayer_channel_for_index(i, j) != 0:
                rgb_channels[0, i, j] = mean_neighbors(bayer_mosaics[0], i, j)
            # Interpolate green
            if bayer_channel_for_index(i, j) != 1:
                rgb_channels[1, i, j] = mean_neighbors_green(bayer_mosaics[1], i, j)
            # Interpolate blue 
            if bayer_channel_for_index(i, j) != 2:
                rgb_channels[2, i, j] = mean_neighbors(bayer_mosaics[2], i, j)

    # Plot interpolated channels
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(rgb_channels[0, :, :])
    axes[1].imshow(rgb_channels[1, :, :])
    axes[2].imshow(rgb_channels[2, :, :])
    plt.show()

    # Combined mosaics into rgb file
    output = np.zeros((height, width, 3))
    output[:, :, 0] = rgb_channels[0, :, :]
    output[:, :, 1] = rgb_channels[1, :, :]
    output[:, :, 2] = rgb_channels[2, :, :]
    output = output.astype(int)

    return output


def mean_neighbors(channel, i, j):
    neighbors = []
    if i - 1 >= 0:
        neighbors.append(channel[i-1, j])
    if i + 1 < channel.shape[0]:
        neighbors.append(channel[i+1, j])
    if j - 1 >= 0:
        neighbors.append(channel[i, j-1])
    if j + 1 < channel.shape[1]:
        neighbors.append(channel[i, j+1])
    if i - 1 >= 0 and j - 1 >= 0:
        neighbors.append(channel[i-1, j-1])
    if i - 1 >= 0 and j + 1 < channel.shape[1]:
        neighbors.append(channel[i-1, j+1])
    if i + 1 < channel.shape[0] and j - 1 >= 0:
        neighbors.append(channel[i+1, j-1])
    if i + 1 < channel.shape[0] and j + 1 < channel.shape[1]:
        neighbors.append(channel[i+1, j+1])

    return np.nanmean(neighbors)
                    

def mean_neighbors_green(channel, i, j):
    neighbors = []
    if i - 1 >= 0:
        neighbors.append(channel[i-1, j])
    if i + 1 < channel.shape[0]:
        neighbors.append(channel[i+1, j])
    if j - 1 >= 0:
        neighbors.append(channel[i, j-1])
    if j + 1 < channel.shape[1]:
        neighbors.append(channel[i, j+1])
    return np.mean(neighbors)
            

"""
Integration
"""
def main():
    img = plt.imread("./data/image1.jpg")
    plt.imshow(img)
    plt.show()
    img_bayer = img_to_cfa(img)
    plt.imshow(img_bayer)
    plt.show()
    output = demosaicing(img_bayer)
    plt.imshow(output)
    plt.show()


if __name__ == "__main__":
    main()
