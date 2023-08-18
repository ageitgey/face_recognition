#!/usr/bin/env python3


# This is a demo of detecting eye status from the users camera. If the users eyes are closed for EYES_CLOSED seconds, the system will start printing out "EYES CLOSED"
# to the terminal until the user presses and holds the spacebar to acknowledge

# this demo must be run with sudo privileges for the keyboard module to work

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# imports
import face_recognition
import cv2
import time
from scipy.spatial import distance as dist

EYES_CLOSED_SECONDS = 5

def main():
    closed_count = 0
    video_capture = cv2.VideoCapture(0)

    ret, frame = video_capture.read(0)
    # cv2.VideoCapture.release()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    process = True

    while True:
        ret, frame = video_capture.read(0)

        # get it into the correct format
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]



        # get the correct face landmarks
        
        if process:
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

            # get eyes
            for face_landmark in face_landmarks_list:
                left_eye = face_landmark['left_eye']
                right_eye = face_landmark['right_eye']


                color = (255,0,0)
                thickness = 2

                cv2.rectangle(small_frame, left_eye[0], right_eye[-1], color, thickness)

                cv2.imshow('Video', small_frame)

                ear_left = get_ear(left_eye)
                ear_right = get_ear(right_eye)

                closed = ear_left < 0.2 and ear_right < 0.2

                if (closed):
                    closed_count += 1

                else:
                    closed_count = 0

                if (closed_count >= EYES_CLOSED_SECONDS):
                    asleep = True
                    while (asleep): #continue this loop until they wake up and acknowledge music
                        print("EYES CLOSED")

                        if cv2.waitKey(1) == 32: #Wait for space key  
                            asleep = False
                            print("EYES OPENED")
                    closed_count = 0

        process = not process
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

def get_ear(eye):

	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

if __name__ == "__main__":
    main()

