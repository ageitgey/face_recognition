Face Recognition
================

The world’s simplest face recognition and face manipulation library.

Built using `dlib`_\ ’s state-of-the-art deep learning face recognition.

| |image0|
| |image1|

Recognize and manipulate faces from Python or from the command line

Free software, MIT license.

Features
--------

Find faces in pictures
^^^^^^^^^^^^^^^^^^^^^^

Find all the faces that appear in a picture.

|image2|

Find and manipulate facial features in pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the locations and outlines of each person’s eyes, nose, mouth and
chin.

|image3|

| Finding facial features is super useful for lots of important stuff.
  But you can also use for really stupid stuff
| like applying `digital make-up`_ (think ‘Meitu’):

|f3|

Identify faces in pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^

Recognize who appears in each photo.

|f4|

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

| Then in you simply run the commnad ``face_recognition``, passing in
| the folder of known people and the folder (or single image) with
  unknown
| people and it tells you who is in each image:

.. code:: bash

    $ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/

    /unknown_pictures/unknown.jpg,Barack Obama
    /face_recognition_test/unknown_pictures/unknown.jpg,unknown_person

| There’s one line in the output for each face. The data is
  comma-separated
| with the filename and the name of the person found.

| An ``unknown_person`` is a face in the image that didn’t match anyone
  in
| your folder of known people.

| If you simply want to know the names of the people in each photograph
  but don’t
| care about file names, you could do this:

.. code:: bash

    $ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

    Barack Obama
    unknown_person

Python API
^^^^^^^^^^

API Docs: [https://face-recognition.readthedocs.io](\ https://fac

.. _dlib: http://dlib.net/
.. _digital make-up: https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py

.. |image0| image:: https://img.shields.io/pypi/v/face_recognition.svg
.. |image1| image:: https://travis-ci.org/ageitgey/face_recognition
.. |image2| image:: https://cloud.githubusercontent.com/assets/896692/23582662/3891b65c-00e4-11e7-848e-0007bca850df.png
.. |image3| image:: https://cloud.githubusercontent.com/assets/896692/23582665/3bbf323c-00e4-11e7-83f9-d42ede9ead2d.png
.. |f3| image:: https://cloud.githubusercontent.com/assets/896692/23582667/3cf969c4-00e4-11e7-82e6-add45ae3992f.png
.. |f4| image:: https://cloud.githubusercontent.com/assets/896692/23582670/405a5268-00e4-11e7-879f-b4de9f727096.png
.. |known| image:: https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png
.. |unknown| image:: https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png
