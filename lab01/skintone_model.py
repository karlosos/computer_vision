# Dla bazy LFW zbudować model koloru skóry i wykorzystać go do segmentacji twarzy
# Wczytać obrazy pojedynczej klasy, np. George Bush
# Dokonać kadrowania do centralnej części obrazu zawierającego twarz
# Przejść do przestrzeni barwnej, np. YCbCb
# Zbudować uśredniony (skumulowany) histogram barw w ww. przestrzeni (dla składowych Cb i Cr, Y pomijamy)
# Znaleźć maksima i marginesy barw wykorzystane jako granice w progowaniu
# Dokonać progowania z granicami obliczonymi wyżej dla obrazów z innej klasy np. Tony Blair

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # Wczytywanie obrazów Busha
    folder = "./data/lfw-deepfunneled/George_W_Bush/"
    files = os.listdir(folder)

    cum_cb_hist = np.zeros((256, 1))
    cum_cr_hist = np.zeros((256, 1))

    for filename in files:
        img = cv2.imread(folder + filename, flags=cv2.IMREAD_COLOR)
        shape = img.shape

        # Kadrowanie centralnej czesci obrazu (20x20 pikseli
        lower_bound = int(shape[0]/2 - 40)
        upper_bound = int(shape[0]/2 + 40)
        cropped_img = img[lower_bound:upper_bound, lower_bound:upper_bound, :] 
        # plt.imshow(cropped_img)
        # plt.show()

        # Przejscie do przestrzeni bartwnej YCbCr
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YCrCb)

        # Obliczenie histogramu dla Cb i Cr
        histSize = 256
        histRange = (0, 256) # the upper boundary is exclusive
        accumulate = False
        cb_hist = cv2.calcHist(cropped_img, [1], None, [256], [0, 256])
        cr_hist = cv2.calcHist(cropped_img, [2], None, [256], [0, 256])
        cum_cb_hist += cb_hist
        cum_cr_hist += cr_hist

    plt.plot(cum_cb_hist)
    plt.plot(cum_cr_hist)
    plt.show()

    # Granice progowania
    max_cr = np.argmax(cum_cr_hist)
    max_cb = np.argmax(cum_cb_hist)
    tol_cr = 10
    tol_cb = 10


    folder = "./data/lfw-deepfunneled/George_W_Bush/"
    files = os.listdir(folder)

    for filename in files:
        img = cv2.imread(folder + filename, flags=cv2.IMREAD_COLOR)
        img_orig = img.copy()
        img_with_mask = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        mask = np.zeros((img.shape[0], img.shape[1]))

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if (np.abs(img[x, y, 1] - max_cr) > tol_cr) & (np.abs(img[x, y, 2] - max_cb) < tol_cb):
                    mask[x, y] = 255
                else:
                    img_with_mask[x, y, 0] = 0
                    img_with_mask[x, y, 1] = 0
                    img_with_mask[x, y, 2] = 0

        cv2.imshow('obraz', img_orig)
        cv2.imshow('mapa', mask)
        cv2.imshow('wynik', img_with_mask)

        k = cv2.waitKey(0)
        if k == 27:
            break

        


if __name__ == "__main__":
    main()
