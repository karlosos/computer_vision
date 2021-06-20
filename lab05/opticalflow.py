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

def ex_01_regular_grid():
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

        quiver_viz(flow, frame2)

    cap.release()
    cv2.destroyAllWindows()


def quiver_viz(flow, frame2):
    X = np.arange(0, flow.shape[1], 10)
    Y = np.arange(0, flow.shape[0], 10)
    U = flow[Y, :, :][:, X, :][:, :, 0]
    V = flow[Y, :, :][:, X, :][:, :, 1]
    fig = plt.figure()
    plt.imshow(frame2)
    plt.quiver(X, Y, U, V)
    plt.draw()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)

def ex_02_sparve_dense():
    """
    Compare sparse and dense optical flow in each channel
    """
    f = "dataset_video.avi"
    cap = cv2.VideoCapture("./data/"+f)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Farneback parameters
    hsv_harneback = np.zeros_like(frame1)
    hsv_harneback[..., 1] = 255

    # L-K parameters
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate Flow farneback
        flow_farneback = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag_farneback, ang_farneback = cv2.cartToPolar(flow_farneback[..., 0], flow_farneback[..., 1])
        hsv_harneback[..., 0] = ang_farneback * 180 / np.pi / 2
        hsv_harneback[..., 2] = cv2.normalize(mag_farneback, None, 0, 255, cv2.NORM_MINMAX)
        rgb_harneback = cv2.cvtColor(hsv_harneback, cv2.COLOR_HSV2BGR)

        # Calculate Flow L-K


        # Finish loop
        prvs = next

        # GUI
        cv2.imshow('rgb_harneback', rgb_harneback)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ex_01_regular_grid()
    ex_02_sparve_dense()
