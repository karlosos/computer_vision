"""
Optical flow
Lab 05

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
https://learnopencv.com/optical-flow-in-opencv/
https://stackoverflow.com/questions/61943240/quiver-plot-with-optical-flow
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from icecream import ic

def main():
    # plik = "dublin.mp4"
    f = "dataset_video.avi"
    cap = cv2.VideoCapture("./data/"+f)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()