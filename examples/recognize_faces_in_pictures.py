import face_recognition
import cv2

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("biden.jpg")
obama_image = face_recognition.load_image_file("obama.jpg")
unknown_image = face_recognition.load_image_file("obama2.jpg")

print("Running the modified version 0.1")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding = face_recognition.face_encodings(biden_image, num_jitters=50, model="large")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

print(biden_face_encoding)
print(biden_face_encoding.shape)
print(unknown_face_encoding.shape)

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
print(results[0])
# print("Is the unknown face a picture of Biden? {}".format(results[0]))
# print("Is the unknown face a picture of Obama? {}".format(results[1]))
# print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

# multiple_faces_image = face_recognition.load_image_file("jane-goodall-monkey-1.jpg")
# faces_locations = face_recognition.face_locations(multiple_faces_image)
# faces_locations_cnn = face_recognition.face_locations(multiple_faces_image, model="cnn")
# print(f"These are the face locations {faces_locations}")
# print(f"These are the face locations found with the refined cnn model {faces_locations_cnn}")

# new_image = cv2.imread("jane-goodall-monkey-1.jpg")
# for face in faces_locations:
#     (top, right, bottom, left) = face
#     cv2.rectangle(new_image, (left, top), (right, bottom), color=(0,255,0), thickness=2)

# # save the image
# cv2.imwrite("boxed_faces.png", new_image)
# print("Boxed Image Created without cnn")

# new_image = cv2.imread("jane-goodall-monkey-1.jpg")
# for face in faces_locations_cnn:
#     (top, right, bottom, left) = face
#     cv2.rectangle(new_image, (left, top), (right, bottom), color=(0,255,0), thickness=2)

# # save the image
# cv2.imwrite("boxed_faces_cnn.png", new_image)
# print("Boxed Image Created with cnn")


# # Display the image
# cv2.imshow('Face Detection', new_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("HERE I GOT")