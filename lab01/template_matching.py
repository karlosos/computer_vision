from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    # Wczytac dane olivetti faces i stworzyc model twarzy (srednia twarz)
    data = fetch_olivetti_faces()
    template = np.zeros((64, 64))

    # Tworzenie wzorca
    for i in range(data['images'].shape[0]):
        img = data['images'][i, :, :]
        template += img

    # Normalizacja wzorca
    template = (template - np.min(template)) / np.max(template)
    div = np.max(template) / float(255)
    template = np.uint8(np.round(template/div))
    w, h = template.shape[::-1]

    plt.imshow(template)
    plt.show()

    # Wczytac obraz do detekcji
    img = cv2.imread('./data/download.jpg', flags=cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Template matching
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)

    # Rysowanie wykrytych miejsc 
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imshow('mapa odpowiedzi', res)
    cv2.imshow('wynik', img)

    k = cv2.waitKey(0)


if __name__ == "__main__":
    main()
