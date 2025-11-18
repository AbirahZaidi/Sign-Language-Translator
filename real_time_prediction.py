import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import pickle
import math

# Load the trained model and label dictionary
model = load_model('test_model.h5')
with open('label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

predicted_word = ""

while True:
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    key = cv2.waitKey(1)
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

        hand_type = hand['type']  # Left or Right

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop safely
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

        # Prepare image for prediction
        imgInput = cv2.resize(imgWhite, (64, 64))
        imgInput = imgInput / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        prediction = model.predict(imgInput)
        predicted_index = np.argmax(prediction)
        predicted_label = label_dict[predicted_index]

        # Display hand type and predicted letter without overlapping
        text_x = x
        text_y_hand = y - 120  # Move this higher for more space (further up)
        text_y_label = y - 20  # Move this higher for the predicted label text

        # Adjusted text positioning and font size to avoid overlap and blurriness
        cv2.putText(img, f'Hand: {hand_type}', (text_x, text_y_hand),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(img, f'{predicted_label}', (text_x, text_y_label - 30),  # Adjusted position
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

        if key == ord('s'):
            predicted_word += predicted_label

    if key == ord('c'):
        predicted_word = ""

    # Show the full word at bottom
    cv2.putText(img, f'Word: {predicted_word}', (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Image", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
