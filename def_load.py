import face_recognition
import cv2
import numpy as np

def img(where):
    print("picture loaded    ")
    image = face_recognition.load_image_file(where)
    face_encoding = face_recognition.face_encodings(image)[0]

    return face_encoding