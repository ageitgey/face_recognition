import numpy as np
import cv2

# Load the image
img = cv2.imread('test_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Load the pre-trained face encoding model
encoding_model = MyFaceEncodingModel()

# Load the pre-stored face encodings and corresponding names
encodings = np.load('face_encodings.npy')
names = np.load('face_names.npy')

# Encode all faces in the input image using the face encoding model
encodings_img = []
for (x, y, w, h) in faces:
    face_gray = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_gray, encoding_model.input_shape[:2])
    encoding = encoding_model.encode(face_resized)
    encodings_img.append(encoding)
encodings_img = np.array(encodings_img)

# Compute the Euclidean distance between the encodings of all detected faces and all stored encodings
distances = np.linalg.norm(encodings - encodings_img[:, np.newaxis], axis=-1)

# Find the closest matching encoding for each detected face
min_distance_indices = np.argmin(distances, axis=0)
min_distances = distances[min_distance_indices, np.arange(len(min_distance_indices))]

# Print the names corresponding to the closest matching encodings
for i, distance in enumerate(min_distances):
    if distance < 0.6:
        print(names[min_distance_indices[i]])
    else:
        print("Unknown")
