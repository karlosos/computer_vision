import numpy as np
import matplotlib.pyplot as plt


def add_padding(img, padding=2):
    padded = np.zeros((img.shape[0] + padding * 2, img.shape[1] + padding * 2))
    padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = img
    return padded


def conv(img, kernel, padding=2):
    padded = add_padding(img, padding)
    output = np.zeros_like(padded)
    (height, width) = padded.shape
    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            c = padded[i-2:i+2+1, j-2:j+2+1]
            output[i, j] = (kernel * c).sum()
    return output[padding:height-padding+1, padding:width-padding+1]


def main():
    img = plt.imread('./data/image1.jpg')
    plt.imshow(img)
    plt.show()

    plt.imshow(img[:, :, 1])
    plt.show()

    kernel = np.full((5, 5), 1)
    output = conv(img[:, :, 1], kernel, padding=2)
    plt.imshow(output)
    plt.show()


if __name__ == '__main__':
    main()