# Face Recognition

The world's simplest face recognition and face manipulation library.

Built using [dlib](http://dlib.net/)'s state-of-the-art deep learning face recognition.

![](https://img.shields.io/pypi/v/face_recognition.svg)
![](https://travis-ci.org/ageitgey/face_recognition)

Recognize faces from Python or from the command line

-   Free software: MIT license
-   Documentation: <https://face-recognition.readthedocs.io>.

## Features

#### Command-Line Interface

##### Identify everyone in a folder full of pictures

When you install `face_recognition`, you get a simple command-line program
called `face_recognition` that you can use to recognize faces in a
photograph or folder full for photographs.

First, you need to provide a folder with one picture of each person you
already know. There should be one image file for each person with the
files named according to who is in the picture:

[img]

Next, you need a second folder with the files you want to identify:

[img]

Then in you simply run the commad `face_recognition`, passing in
the folder of known people and the folder (or single image) with unknown
people:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/



```

#### Python API

##### Automatically find all the faces in an image

##### Automatically locate the facial features of a person in an image

##### Recognize faces in images and identify who they are

## Examples

All the examples are available [here](https://github.com/ageitgey/face_recognition/tree/master/examples).

* [Find faces in an photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [Identify specific facial features in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Apply (horribly ugly) digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)
* [Find and recognize unknown faces in a photograph based on photographs of known people](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)

## Thanks

* Many, many thanks to [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom))
  for creating dlib and for providing the trained facial feature detection and face encoding models
  used in this library.
* Everyone who works on all the awesome Python data science libraries like numpy, scipy, scikit-image,
  pillow, etc, etc that makes this kind of stuff so easy and fun in Python.
* [Cookiecutter](https://github.com/audreyr/cookiecutter) and the
  [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template
  for making Python project packaging way more tolerable.
