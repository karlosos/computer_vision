import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt


def main():
    f = "same_size.mp4"
    cap = cv2.VideoCapture("./data/" + f)
    frame = load_frame(cap)

    # Select object
    r = cv2.selectROI("frame", frame)
    ic(r)
    cropped = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    # Color model for tracked object
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    # Histogram for hue with 20 bins, values from 0 to 180
    hist = cv2.calcHist([hsv], [0], None, [20], [0, 180]).reshape(-1)
    hist = hist/np.sum(hist)
    ic(hist)
    plt.bar(range(len(hist)), hist)
    plt.show()

    # Video
    while (1):
        frame = load_frame(cap)

        cv2.imshow('frame', frame)
        # k = cv2.waitKey(30) & 0xff
        k = cv2.waitKey(0)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def load_frame(cap):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 480))
    return frame


if __name__ == '__main__':
    main()