from __future__ import print_function
import cv2 as cv
import matplotlib.pyplot as plt


def detectAndDisplay(img, cascade):
    frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    balls = cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in balls:
        center = (x + w//2, y + h//2)
        img = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        ballROI = frame_gray[y:y+h,x:x+w]
    return img


def main():
    cascade = cv.CascadeClassifier()

    #-- 1. Load the cascade
    if not cascade.load(cv.samples.findFile('./cascade/cascade.xml')):
        print('--(!)Error loading face cascade')
        exit(0)

    #-- 2. Read the image
    for i in range(20):
        img = cv.imread(f'./dataset/tests/ball({201+i}).jpg')
        img = detectAndDisplay(img, cascade)
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    main()
