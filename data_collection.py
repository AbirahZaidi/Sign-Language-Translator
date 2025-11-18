import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

folder = f"C:/Users/abira/OneDrive/Desktop/Sign Language Detection/Data/Z"

if not os.path.exists(folder):
    
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region safely
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Cropped", imgCrop)
        cv2.imshow("White", imgWhite)

    cv2.imshow("Webcam", img)

    key = cv2.waitKey(1)

    if key == ord('s') and hands:
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        counter += 1
        print(f"Image {counter} saved: {filename}")

    elif key == ord('q'):
        print("Closing camera...")
        break

cap.release()
cv2.destroyAllWindows()