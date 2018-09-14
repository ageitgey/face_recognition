# -*- coding: utf-8 -*-
#######################################
__file__ = "facerec_with_gui.py"
__author__ = "Mesut Pişkin"
__version__ = "1.0"
__email__ = "mesutpiskin@outlook.com"
#######################################

'''

Create a faces folder for your face photos and copy the photos into this directory. 

Photo names format; 
mesut_1.jpg 
yourname_1.png 
mesut_2.jpeg 
mustafa_1.png
alis_1.jpg

should be in the format.

'''


from tkinter import *
import PIL.Image
import PIL.ImageTk
import tkinter.filedialog
import face_recognition
import cv2
import os
import threading

# Form object
frame = Tk()
frame.resizable(width=FALSE, height=FALSE)
frame.title("Face Recognition GUI")
frame.geometry("1024x768")


global lblImage
video_capture = cv2.VideoCapture(0)
known_face_encodings = []
known_face_names = []
global camera_is_open
global btnOpenCamera


def trainFaces():
    print("---- Training Started ----")
    for root, dirs, files in os.walk("./faces"):
        for filename in files:
            file_result = filename.split("_")
            known_face_names.append(file_result[0])
            image = face_recognition.load_image_file("faces/"+filename)
            image_face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(image_face_encoding)
            print("Name: " + file_result[0])
    print("---- Training Completed ----")


def faceRecognitionFromPicture(cvframe):
    print("---- Recognized Started ----")
    small_frame = cv2.resize(cvframe, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    small_rgb_frame = small_frame[:, :, ::-1]

    # get face location
    face_locations = face_recognition.face_locations(small_rgb_frame)
    print("- Face location scan completed")

    face_encodings = face_recognition.face_encodings(
        small_rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)
        name = "not recognized"  # default name is not recognized

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # print face data
    print(*face_locations, sep='\n')
    print(*face_names, sep='\n')
    print("- Face name searching completed")
    # draw face rectangle and name on current frame
    drawFaceOnImage(cvframe, face_locations, face_names)
    # Label string
    faceNames = ''.join(face_names)
    count = str(len(face_locations))
    location = ','.join([str(i) for i in face_locations])
    return_string = "\nNames: "+faceNames + \
        "\nFace Count: "+count+"\nLocations: "+location+"\n"
    lblTag["text"] = return_string
    print("---- Recognized Completed ----")


def drawFaceOnImage(frame, face_locations, face_names):

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (153, 0, 51), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (153, 0, 51), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 2)
    # write temp image file for lblimage item
    cv2.imwrite("temp.jpg", frame)


def openFile():
    camera_is_open = False
    # open file dialog for picture
    filename = tkinter.filedialog.askopenfilename(
        initialdir="/", title="Choose Photo")
    # recognize face
    cvframe = cv2.imread(filename)
    faceRecognitionFromPicture(cvframe)

    # get recognized picture
    im = PIL.Image.open("temp.jpg")
    im = im.resize((700, 400))
    photo = PIL.ImageTk.PhotoImage(im)
    lblImage.configure(image=photo)
    lblImage.image = photo


def openCamera():
    global btnOpenCamera

    global camera_is_open
    if camera_is_open == False:
        camera_is_open = True
        btnOpenCamera["text"] = "Stop Camera"
        videoThread = threading.Thread(
            target=processCameraFrameForTkinter, args=())
        videoThread.start()
    else:
        camera_is_open = False
        btnOpenCamera["text"] = "Start Camera"


def processCameraFrameForTkinter():
    global camera_is_open
    while camera_is_open:
        ret, frame = video_capture.read()
        faceRecognitionFromPicture(frame)
        # get recognized picture
        im = PIL.Image.open("temp.jpg")
        im = im.resize((960, 540))
        photo = PIL.ImageTk.PhotoImage(im)
        lblImage.configure(image=photo)
        lblImage.image = photo
        # or use cv2.imshow()


# train face in faces folder
trainFaces()
# form Components
btnOpenFile = Button(text="Recognize from photos", command=openFile)
lblTag = Label(text="",
               bg="red", fg="white", font="Arial 18")
lblImage = Label()
btnOpenCamera = Button(text="Recognize from camera", command=openCamera)

btnOpenFile.pack()
btnOpenCamera.pack()
lblTag.pack()
lblImage.pack()
camera_is_open = False
mainloop()
