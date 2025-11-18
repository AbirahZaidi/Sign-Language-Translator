import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Path to the Data folder
path = 'C:/Users/abira/OneDrive/Desktop/Sign Language Detection/Data'

images = []
labels = []
label_dict = {}  # To store A=0, B=1, C=2, etc.

# Assign numeric labels to A-Z
for idx, folder_name in enumerate(sorted(os.listdir(path))):
    label_dict[idx] = folder_name
    folder_path = os.path.join(path, folder_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Resize all images
        images.append(img)
        labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0

# One hot encode labels
labels = to_categorical(labels, num_classes=26)

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Save the model
model.save('model.h5')

# Save label dictionary (for decoding later)
import pickle
with open('label_dict.pkl', 'wb') as f:
    pickle.dump(label_dict, f)
   
# Evaluate model
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
target_names = [label_dict[i] for i in range(26)]
print(classification_report(y_true, y_pred, target_names=target_names))

print("Training completed and model saved!")