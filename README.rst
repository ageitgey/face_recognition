Face Recognition
================

| Recognize and manipulate faces from Python or from the command line
  with
| the world's simplest face recognition library.

| Built using `dlib <http://dlib.net/>`__'s state-of-the-art face
  recognition
| built with deep learning. The model has an accuracy of 99.38% on the
| `Labeled Faces in the Wild <http://vis-www.cs.umass.edu/lfw/>`__
  benchmark.

| This also provides a simple ``face_recognition`` command line tool
  that lets
| you do face recognition on a folder of images from the command line!

| |PyPI|
| |Build Status|
| |Documentation Status|

Features
--------

Find faces in pictures
^^^^^^^^^^^^^^^^^^^^^^

Find all the faces that appear in a picture:

|image3|

.. code:: python

    import face_recognition
    image = face_recognition.load_image_file("your_file.jpg")
    face_locations = face_recognition.face_locations(image)

Find and manipulate facial features in pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the locations and outlines of each person's eyes, nose, mouth and
chin.

|image4|

.. code:: python

    import face_recognition
    image = face_recognition.load_image_file("your_file.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)

| Finding facial features is super useful for lots of important stuff.
  But you can also use for really stupid stuff
| like applying `digital
  make-up <https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py>`__
  (think 'Meitu'):

|image5|

Identify faces in pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^

Recognize who appears in each photo.

|image6|

.. code:: python

    import face_recognition
    known_image = face_recognition.load_image_file("biden.jpg")
    unknown_image = face_recognition.load_image_file("unknown.jpg")

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

Installation
------------

| Python 3 / Python 2 are fully supported. Only macOS and
| Linux are tested. I have no idea if this will work on Windows.

Install this module from pypi using ``pip3`` (or ``pip2`` for Python 2):

.. code:: bash

    pip3 install face_recognition

| IMPORTANT NOTE: It's very likely that you will run into problems when
  pip tries to compile
| the ``dlib`` dependency. If that happens, check out this guide to
  installing
| dlib from source (instead of from pip) to fix the error:

`How to install dlib from
source <https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf>`__

| After manually installing ``dlib``, try running
  ``pip3 install face_recognition``
| again to complete your installation.

Usage
-----

Command-Line Interface
^^^^^^^^^^^^^^^^^^^^^^

| When you install ``face_recognition``, you get a simple command-line
  program
| called ``face_recognition`` that you can use to recognize faces in a
| photograph or folder full for photographs.

| First, you need to provide a folder with one picture of each person
  you
| already know. There should be one image file for each person with the
| files named according to who is in the picture:

|known|

Next, you need a second folder with the files you want to identify:

|unknown|

| Then in you simply run the command ``face_recognition``, passing in
| the folder of known people and the folder (or single image) with
  unknown
| people and it tells you who is in each image:

.. code:: bash

    $ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/

    /unknown_pictures/unknown.jpg,Barack Obama
    /face_recognition_test/unknown_pictures/unknown.jpg,unknown_person

| There's one line in the output for each face. The data is
  comma-separated
| with the filename and the name of the person found.

| An ``unknown_person`` is a face in the image that didn't match anyone
  in
| your folder of known people.

| If you simply want to know the names of the people in each photograph
  but don't
| care about file names, you could do this:

.. code:: bash

    $ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

    Barack Obama
    unknown_person

Python Module
^^^^^^^^^^^^^

| You can import the ``face_recognition`` module and then easily
  manipulate
| faces with just a couple of lines of code. It's super easy!

API Docs:
`https://face-recognition.readthedocs.io <https://face-recognition.readthedocs.io/en/latest/face_recognition.html>`__.

Automatically find all the faces in an image
''''''''''''''''''''''''''''''''''''''''''''

.. code:: python

    import face_recognition

    image = face_recognition.load_image_file("my_picture.jpg")
    face_locations = face_recognition.face_locations(image)

    # face_locations is now an array listing the co-ordinates of each face!

| See `this
  example <https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py>`__
| to try it out.

Automatically locate the facial features of a person in an image
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: python

    import face_recognition

    image = face_recognition.load_image_file("my_picture.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)

    # face_landmarks_list is now an array with the locations of each facial feature in each face.
    # face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.

| See `this
  example <https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py>`__
| to try it out.

Recognize faces in images and identify who they are
'''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: python

    import face_recognition

    picture_of_me = face_recognition.load_image_file("me.jpg")
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

    unknown_picture = face_recognition.load_image_file("unknown.jpg")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

    # Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

    if results[0] == True:
        print("It's a picture of me!")
    else:
        print("It's not a picture of me!")

| See `this
  example <https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py>`__
| to try it out.

Python Code Examples
--------------------

All the examples are available
`here <https://github.com/ageitgey/face_recognition/tree/master/examples>`__.

-  `Find faces in a
   photograph <https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py>`__
-  `Identify specific facial features in a
   photograph <https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py>`__
-  `Apply (horribly ugly) digital
   make-up <https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py>`__
-  `Find and recognize unknown faces in a photograph based on
   photographs of known
   people <https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py>`__
-  `Recognize faces in live video using your webcam (Requires OpenCV to
   be
   installed) <https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py>`__

How Face Recognition Works
--------------------------

| If you want to learn how face location and recognition work instead of
| depending on a black box library, `read my
  article <https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78>`__.

Caveats
-------

-  The face recognition model is trained on adults and does not work
   very well on children. It tends to mix
   up children quite easy using the default comparison threshold of 0.6.

Common Issues
-------------

Issue: ``Illegal instruction (core dumped)`` when using face\_recognition or running examples.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

| Solution: ``dlib`` is compiled with SSE4 or AVX support, but your CPU
  is too old and doesn't support that.
| You'll need to recompile ``dlib`` after `making the code change
  outlined
  here <https://github.com/ageitgey/face_recognition/issues/11#issuecomment-287398611>`__.

Issue: ``RuntimeError: Unsupported image type, must be 8bit gray or RGB image.`` when running the webcam example.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Solution: Your webcam probably isn't set up correctly with OpenCV. `Look
here for
more <https://github.com/ageitgey/face_recognition/issues/21#issuecomment-287779524>`__.

Issue: ``MemoryError`` when running ``pip2 install face_recognition``
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

| Solution: The face\_recognition\_models file is too big for your
  available pip cache memory. Instead,
| try ``pip2 --no-cache-dir install face_recognition`` to avoid the
  issue.

Thanks
------

-  Many, many thanks to `Davis King <https://github.com/davisking>`__
   (`@nulhom <https://twitter.com/nulhom>`__)
   for creating dlib and for providing the trained facial feature
   detection and face encoding models
   used in this library. For more information on the ResNet that powers
   the face encodings, check out
   his `blog
   post <http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html>`__.
-  Thanks to everyone who works on all the awesome Python data science
   libraries like numpy, scipy, scikit-image,
   pillow, etc, etc that makes this kind of stuff so easy and fun in
   Python.
-  Thanks to `Cookiecutter <https://github.com/audreyr/cookiecutter>`__
   and the
   `audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`__
   project template
   for making Python project packaging way more tolerable.

.. |PyPI| image:: https://img.shields.io/pypi/v/face_recognition.svg
   :target: https://pypi.python.org/pypi/face_recognition
.. |Build Status| image:: https://travis-ci.org/ageitgey/face_recognition.svg?branch=master
   :target: https://travis-ci.org/ageitgey/face_recognition
.. |Documentation Status| image:: https://readthedocs.org/projects/face-recognition/badge/?version=latest
   :target: http://face-recognition.readthedocs.io/en/latest/?badge=latest
.. |image3| image:: https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png
.. |image4| image:: https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png
.. |image5| image:: https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png
.. |image6| image:: https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png
.. |known| image:: https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png
.. |unknown| image:: https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png

