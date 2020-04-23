import face_recognition
import numpy as np
import argparse
import cv2

# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-bs", "--blur-shape", type=str, default="circle",
	help="shape of blur")   
args = vars(ap.parse_args())

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="cnn")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if args["blur_shape"] == "square" :
            # Extract the region of the image that contains the face
            face_image = frame[top:bottom, left:right]

            # Blur the face image
            face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

            # Put the blurred face region back into the frame image
            frame[top:bottom, left:right] = face_image
        else:
            x,y = (int((right + left)/2), int((top + bottom)/2))

            #After experimenting, radius of  0.75 * max(boxHeigh, bohWidth) seems to cover most
            #of the face
            radius = max(abs(top-bottom) * 3/4, abs(right-left)*3/4)

            #Create a mask for gaussian blur
            mask = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)
            mask = cv2.circle(mask, (x, y), int(radius), (255, 255, 255), -1)
        
            #Blur the image
            blurred_image = cv2.GaussianBlur(frame, (99, 99), 30)
        
            #Replace the face with blurred face
            frame = np.where(mask!=[255, 255, 255], frame, blurred_image)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
