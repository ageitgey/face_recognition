import face_recognition
import cv2
import numpy as np
import def_load as load
import random as rn

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
print("GO!")
# for a in range(10):
#     print("trying", a)
#     video_capture = cv2.VideoCapture(a, cv2.CAP_DSHOW)
#     ret, frame = video_capture.read()
#     print(ret)
#     if ret == True:
#         break
#     else:
#         video_capture.release()
video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)


print("video captured")
# print(video_capture)

# Load a sample picture and learn how to recognize it.


IIB8PolednovaSofie = load.img("faces/IIB8PolednovaSofie.png")
IIIB8AmbrosMichael = load.img("faces/IIIB8AmbrosMichael.png")
IIIB8AxmanovaPavlina = load.img("faces/IIIB8AxmanovaPavlina.png")
IIIB8BuresEBenjamin = load.img("faces/IIIB8BuresEBenjamin.png")
IIIB8CoufalovaNoemi = load.img("faces/IIIB8CoufalovaNoemi.png")
IIIB8DolezalovaHelena = load.img("faces/IIIB8DolezalovaHelena.png")
IIIB8HavrlantSimon = load.img("faces/IIIB8HavrlantSimon.png")
IIIB8HimrJonas = load.img("faces/IIIB8HimrJonas.png")
IIIB8HockadaySMatthew = load.img("faces/IIIB8HockadaySMatthew.png")
IIIB8HoudekJan = load.img("faces/IIIB8HoudekJan.png")
IIIB8JuraskovaAneta = load.img("faces/IIIB8JuraskovaAneta.png")
IIIB8KlimkovaViola = load.img("faces/IIIB8KlimkovaViola.png")
IIIB8KolarJan = load.img("faces/IIIB8KolarJan.png")
IIIB8KonecnyPetr = load.img("faces/IIIB8KonecnyPetr.png")
IIIB8LazarovaSofie = load.img("faces/IIIB8LazarovaSofie.png")
IIIB8MadrovaGabriela = load.img("faces/IIIB8MadrovaGabriela.png")
IIIB8PaulickovaTereza = load.img("faces/IIIB8PaulickovaTereza.png")
IIIB8PetrivalskyNil = load.img("faces/IIIB8PetrivalskyNil.png")
IIIB8RatajovaTerezie = load.img("faces/IIIB8RatajovaTerezie.png")
IIIB8SaradinMatej = load.img("faces/IIIB8SaradinMatej.png")
IIIB8SimackovaMEmma = load.img("faces/IIIB8SimackovaMEmma.png")
IIIB8SitovaHana = load.img("faces/IIIB8SitovaHana.png")
IIIB8SmutnaHana = load.img("faces/IIIB8SmutnaHana.png")
IIIB8StefanovaAnna = load.img("faces/IIIB8StefanovaAnna.png")
IIIB8StyblovaTatiana = load.img("faces/IIIB8StyblovaTatiana.png")
IIIB8TomeckovaAnna = load.img("faces/IIIB8TomeckovaAnna.png")
IIIB8VaclavikovaAnna = load.img("faces/IIIB8VaclavikovaAnna.png")
IIIB8VeseckyJakub = load.img("faces/IIIB8VeseckyJakub.png")
IIIB8ZidkovaLinda = load.img("faces/IIIB8ZidkovaLinda.png")
print("pictures loaded")

known_face_encodings = [IIB8PolednovaSofie, IIIB8AmbrosMichael, IIIB8AxmanovaPavlina, IIIB8BuresEBenjamin, IIIB8CoufalovaNoemi, IIIB8DolezalovaHelena, IIIB8HavrlantSimon, IIIB8HimrJonas, IIIB8HockadaySMatthew, IIIB8HoudekJan, IIIB8JuraskovaAneta, IIIB8KlimkovaViola, IIIB8KolarJan, IIIB8KonecnyPetr, IIIB8LazarovaSofie, IIIB8MadrovaGabriela, IIIB8PaulickovaTereza, IIIB8PetrivalskyNil, IIIB8RatajovaTerezie, IIIB8SaradinMatej, IIIB8SimackovaMEmma, IIIB8SitovaHana, IIIB8SmutnaHana, IIIB8StefanovaAnna, IIIB8StyblovaTatiana, IIIB8TomeckovaAnna, IIIB8VaclavikovaAnna, IIIB8VeseckyJakub, IIIB8ZidkovaLinda]
known_face_names = ['IIB8PolednovaSofie', 'IIIB8AmbrosMichael', 'IIIB8AxmanovaPavlina', 'IIIB8BuresEBenjamin', 'IIIB8CoufalovaNoemi', 'IIIB8DolezalovaHelena', 'IIIB8HavrlantSimon', 'IIIB8HimrJonas', 'IIIB8HockadaySMatthew', 'IIIB8HoudekJan', 'IIIB8JuraskovaAneta', 'IIIB8KlimkovaViola', 'IIIB8KolarJan', 'IIIB8KonecnyPetr', 'IIIB8LazarovaSofie', 'IIIB8MadrovaGabriela', 'IIIB8PaulickovaTereza', 'IIIB8PetrivalskyNil', 'IIIB8RatajovaTerezie', 'IIIB8SaradinMatej', 'IIIB8SimackovaMEmma', 'IIIB8SitovaHana', 'IIIB8SmutnaHana', 'IIIB8StefanovaAnna', 'IIIB8StyblovaTatiana', 'IIIB8TomeckovaAnna', 'IIIB8VaclavikovaAnna', 'IIIB8VeseckyJakub', 'IIIB8ZidkovaLinda']



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
print("variables")
loop_noob = 0
while True:
    loop_noob = loop_noob + 1
    # Grab a single frame of video
    ret, frame = video_capture.read()
    print(ret, loop_noob)

    # Resize frame of video to 1/4 size for faster face recognition processing
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    # small_frame = frame

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
