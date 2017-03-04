# Face Recognition

The world's simplest face recognition and face manipulation library.

Built using [dlib](http://dlib.net/)'s state-of-the-art deep learning face recognition.

![](https://img.shields.io/pypi/v/face_recognition.svg)
![](https://travis-ci.org/ageitgey/face_recognition)

Recognize and manipulate faces from Python or from the command line

Free software, MIT license.

## Features

#### Find faces in pictures

Find all the faces that appear in a picture.

![](https://cloud.githubusercontent.com/assets/896692/23582662/3891b65c-00e4-11e7-848e-0007bca850df.png)

#### Find and manipulate facial features in pictures

Get the locations and outlines of each person's eyes, nose, mouth and chin.

![](https://cloud.githubusercontent.com/assets/896692/23582665/3bbf323c-00e4-11e7-83f9-d42ede9ead2d.png)

Finding facial features is super useful for lots of important stuff. But you can also use for really stupid stuff
like applying [digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (think 'Meitu'):

![f3](https://cloud.githubusercontent.com/assets/896692/23582667/3cf969c4-00e4-11e7-82e6-add45ae3992f.png)

#### Identify faces in pictures

Recognize who appears in each photo.

![f4](https://cloud.githubusercontent.com/assets/896692/23582670/405a5268-00e4-11e7-879f-b4de9f727096.png)

## Usage

#### Command-Line Interface

When you install `face_recognition`, you get a simple command-line program
called `face_recognition` that you can use to recognize faces in a
photograph or folder full for photographs.

First, you need to provide a folder with one picture of each person you
already know. There should be one image file for each person with the
files named according to who is in the picture:

![known](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

Next, you need a second folder with the files you want to identify:

![unknown](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

Then in you simply run the commnad `face_recognition`, passing in
the folder of known people and the folder (or single image) with unknown
people and it tells you who is in each image:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

There's one line in the output for each face. The data is comma-separated
with the filename and the name of the person found.

An `unknown_person` is a face in the image that didn't match anyone in
your folder of known people.

If you simply want to know the names of the people in each photograph but don't
care about file names, you could do this:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

Barack Obama
unknown_person
```


#### Python API

API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io).

You can import the `face_recognition` module and then easily manipulate
faces with just a couple of lines of code. It's super easy!

##### Automatically find all the faces in an image

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
 to try it out.

##### Automatically locate the facial features of a person in an image

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

# face_landmarks_list is now an array with the locations of each facial feature in each face.
# face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
 to try it out.

##### Recognize faces in images and identify who they are

```python
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
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
 to try it out.


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
