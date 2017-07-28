=====
Usage
=====

To use Face Recognition in a project::

    import face_recognition

See the examples in the /examples folder on github for how to use each function.

You can also check the API docs for the 'face_recognition' module to see the possible parameters for each function.

The basic idea is that first you load an image::

    import face_recognition

    image = face_recognition.load_image_file("your_file.jpg")

That loads the image into a numpy array. If you already have an image in a numpy array, you can skip this step.

Then you can perform operations on the image, like finding faces, identifying facial features or finding face encodings::

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    # Or maybe find the facial features in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    # Or you could get face encodings for each face in the image:
    list_of_face_encodings = face_recognition.face_encodings(image)

Face encodings can be compared against each other to see if the faces are a match. Note: Finding the encoding for a face
is a bit slow, so you might want to save the results for each image in a database or cache if you need to refer back to
it later.

But once you have the encodings for faces, you can compare them like this::

    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = face_recognition.compare_faces(known_face_encodings, a_single_unknown_face_encoding)

It's that simple! Check out the examples for more details.
