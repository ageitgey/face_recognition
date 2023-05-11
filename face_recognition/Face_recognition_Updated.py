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

# Loop through each detected face
for (x, y, w, h) in faces:
    # Extract the face ROI from the grayscale image
    face_gray = gray[y:y+h, x:x+w]
    # Resize the face ROI to match the expected input size of the face encoding model
    face_resized = cv2.resize(face_gray, encoding_model.input_shape[:2])
    # Encode the face ROI using the face encoding model
    encoding = encoding_model.encode(face_resized)
    # Compute the Euclidean distance between the encoding of the detected face and all stored encodings
    distances = np.linalg.norm(encodings - encoding, axis=1)
    min_distance_index = np.argmin(distances)
    # Check if the closest match is below a certain threshold (e.g., 0.6)
    if distances[min_distance_index] < 0.6:
        print(names[min_distance_index])
    else:
        print("Unknown")
