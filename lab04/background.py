"""
TODO: 
1: Zmodyfikować fragment odpowiadający za odejmowanie tła, w taki sposób aby odejmować ruchomą średnią z kilkunastu klatek (w tej wersji odejmowana jest tylko pierwsza klatka) przy uwzględnieniu stałej zapominania alfa (patrz wykład) - porównać z wynikiem działania modelu MOG2 (Mixture of Gaussians), wyciągnąć wnioski dotyczące liczby klatek w ruchomej średniej i wartości alfa
2. Dodać element usuwania cieni, tak jak zaprezentowano na wykładzie (za pomocą nowej maski, która analizuje różnice w przestrzeni HSV) - zaproponować progi alfa, beta, tau h i tał s
"""


import cv2
import numpy as np


def main():
    kat = "./data"

    plik = "dublin.mp4"
    # plik = "dataset_video.avi"
    cap = cv2.VideoCapture(kat + "/" + plik)

    fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2()

    _, frame = cap.read()
    first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
    previous_background = first_gray
    alpha = 0.05
    # Przy małej alfa jest długa ścieżka (trailing)
    last_frames = [frame, ]
    num_last_frames = 24

    while 1:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        background = (1 - alpha)*previous_background + alpha*gray_frame
        background = background.astype('uint8')
        fgmask = cv2.absdiff(background, gray_frame)
        _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
        previous_background = background

        # noisefld=np.random.randn(frame.shape[0],frame.shape[1])
        # frame[:,:,0]=(frame[:,:,0]+10*noisefld).astype('int')
        # frame[:,:,1]=(frame[:,:,1]+10*noisefld).astype('int')
        # frame[:,:,2]=(frame[:,:,2]+10*noisefld).astype('int')

        fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(frame)
        # HSV shadows
        I = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if len(last_frames) >= num_last_frames:
            last_frames.pop(0)
        last_frames.append(frame)
        B = np.mean(last_frames, axis=0).astype('uint8')
        a = 0.1
        b = 0.2
        th = 0.2
        ts = 0.2
        cond_1 = I[:, :, 2]/B[:, :, 2] >= a
        cond_2 = I[:, :, 2]/B[:, :, 2] <= b
        cond_3 = I[:, :, 1] - B[:, :, 1] <= ts
        cond_4 = np.abs(I[:, :, 0] - B[:, :, 0]) <= th
        SP = np.logical_and(np.logical_and(cond_1, cond_2), np.logical_and(cond_3, cond_4))
        SP = SP.astype('uint8')*255

        cv2.namedWindow("Background Subtraction", 0)
        cv2.namedWindow("Background Subtraction Adaptive Gaussian", 0)
        cv2.namedWindow("Shadow", 0)
        cv2.namedWindow("Original", 0)

        cv2.imshow("Background Subtraction", fgmask)
        cv2.imshow("Background Subtraction Adaptive Gaussian", fgbgAdaptiveGaussainmask)
        cv2.imshow("Original", frame)
        cv2.imshow("Shadow", SP)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program Closed")


if __name__ == "__main__":
    main()
