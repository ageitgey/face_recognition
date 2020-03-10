"""
This is an example of using alternative face landmarks detector.

When should I use this example?
This example is useful when you wish to use alternative face landmarks detector,
e.g. you want recognize not only strcitly frontal faces.
(default face landmarks detector give satisfactory results only with frontal faces)

NOTE: This example requires face_alignment to be installed! You can install it with pip:

$ pip3 install face_alignment 

"""

import face_recognition
import face_alignment

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("biden.jpg")
obama_image = face_recognition.load_image_file("obama.jpg")
unknown_image = face_recognition.load_image_file("obama2.jpg")

# prepare alternative landmarks detector
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                  device='cpu',
                                  flip_input=True)

# helper method for boxes conversion
def aligner_boxes(boxes):
    return [(left, top, right, bottom)
            for top, right, bottom, left in boxes]

# encode end get landmarks by mean of alternative landmarks detector
def face_encodings_and_landmarks(image):
    # detect face locations
    locations = face_recognition.face_locations(image, model="cnn")

    # detect face landmarks
    landmark_points = [
        lp.astype(int).tolist()
        for lp in fa.get_landmarks_from_image(image, aligner_boxes(locations))]

    # encode faces
    encodings = face_recognition.face_encodings(
        image, known_face_locations=locations,
        landmark_points=landmark_points)

    # encode landmarks
    landmarks = face_recognition.face_landmarks(
        landmark_points=landmark_points)

    return encodings, landmarks


# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding = face_encodings_and_landmarks(biden_image)[0][0]
    obama_face_encoding = face_encodings_and_landmarks(obama_image)[0][0]
    unknown_face_encoding = face_encodings_and_landmarks(unknown_image)[0][0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Biden? {}".format(results[0]))
print("Is the unknown face a picture of Obama? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(
    True not in results))
