import numpy as np
import cv2

img = cv2.imread('test_image.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

encodings = np.load('face_encodings.npy')
names = np.load('face_names.npy')

for (x, y, w, h) in faces:
    face_gray = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_gray, encoding_model.input_shape[:2])
    encoding = encoding_model.encode(face_resized)
    distances = np.linalg.norm(encodings - encoding, axis=1)
    min_distance_index = np.argmin(distances)
    if distances[min_distance_index] < 0.6:
        print(names[min_distance_index])
    else:
        print("Unknown")
