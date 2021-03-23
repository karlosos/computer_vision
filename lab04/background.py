"""
TODO: 
1: Zmodyfikować fragment odpowiadający za odejmowanie tła, w taki sposób aby odejmować ruchomą średnią z kilkunastu klatek (w tej wersji odejmowana jest tylko pierwsza klatka) przy uwzględnieniu stałej zapominania alfa (patrz wykład) - porównać z wynikiem działania modelu MOG2 (Mixture of Gaussians), wyciągnąć wnioski dotyczące liczby klatek w ruchomej średniej i wartości alfa
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
    last_frames = [first_gray, ]
    previous_background = first_gray
    num_last_frames = 24
    alpha = 0.05
    # Przy małej alfa jest długa ścieżka (trailing)

    while 1:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if (len(last_frames) >= num_last_frames):
        #     last_frames.pop(0)
        # last_frames.append(gray_frame)
        # weights = np.logspace(0, 1, len(last_frames))
        # mean_frame = np.average(last_frames, axis=0, weights=weights).astype('uint8')
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

        cv2.namedWindow("Background Subtraction", 0)
        cv2.namedWindow("Background Subtraction Adaptive Gaussian", 0)
        cv2.namedWindow("Original", 0)

        cv2.imshow("Background Subtraction", fgmask)
        cv2.imshow("Background Subtraction Adaptive Gaussian", fgbgAdaptiveGaussainmask)
        cv2.imshow("Original", frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program Closed")


if __name__ == "__main__":
    main()
