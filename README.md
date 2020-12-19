# Face Recognition

_You can also read a translated version of this file [in Chinese 简体中文版](https://github.com/ageitgey/face_recognition/blob/master/README_Simplified_Chinese.md) or [in Korean 한국어](https://github.com/ageitgey/face_recognition/blob/master/README_Korean.md) or [in Japanese 日本語](https://github.com/m-i-k-i/face_recognition/blob/master/README_Japanese.md)._

Recognize and manipulate faces from Python or from the command line with
the world's simplest face recognition library.

Built using [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.

This also provides a simple `face_recognition` command line tool that lets
you do face recognition on a folder of images from the command line!


[![PyPI](https://img.shields.io/pypi/v/face_recognition.svg)](https://pypi.python.org/pypi/face_recognition)
[![Build Status](https://github.com/ageitgey/face_recognition/workflows/CI/badge.svg?branch=master&event=push)](https://github.com/ageitgey/face_recognition/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)

## Features

#### Find faces in pictures

Find all the faces that appear in a picture:

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
```

#### Find and manipulate facial features in pictures

Get the locations and outlines of each person's eyes, nose, mouth and chin.

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

Finding facial features is super useful for lots of important stuff. But you can also use it for really stupid stuff
like applying [digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (think 'Meitu'):

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### Identify faces in pictures

Recognize who appears in each photo.

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```

You can even use this library with other Python libraries to do real-time face recognition:

![](https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif)

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) for the code.

## Online Demos

User-contributed shared Jupyter notebook demo (not officially supported): [![Deepnote](https://beta.deepnote.org/buttons/try-in-a-jupyter-notebook.svg)](https://beta.deepnote.org/launch?template=face_recognition)

## Installation

### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux (Windows not officially supported, but might work)

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have dlib already installed with Python bindings:

  * [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
  
Then, make sure you have cmake installed:  
 
```brew install cmake```

Finally, install this module from pypi using `pip3` (or `pip2` for Python 2):

```bash
pip3 install face_recognition
```

Alternatively, you can try this library with [Docker](https://www.docker.com/), see [this section](#deployment).

If you are having trouble with installation, you can also try out a
[pre-configured VM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).

#### Installing on an Nvidia Jetson Nano board

 * [Jetson Nano installation instructions](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd)
   * Please follow the instructions in the article carefully. There is current a bug in the CUDA libraries on the Jetson Nano that will cause this library to fail silently if you don't follow the instructions in the article to comment out a line in dlib and recompile it.

#### Installing on Raspberry Pi 2+

  * [Raspberry Pi 2+ installation instructions](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

#### Installing on FreeBSD

```bash
pkg install graphics/py-face_recognition
```

#### Installing on Windows

While Windows isn't officially supported, helpful users have posted instructions on how to install this library:

  * [@masoudr's Windows 10 installation guide (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

#### Installing a pre-configured Virtual Machine image

  * [Download the pre-configured VM image](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b) (for VMware Player or VirtualBox).

## Usage

### Command-Line Interface

When you install `face_recognition`, you get two simple command-line 
programs:

* `face_recognition` - Recognize faces in a photograph or folder full for 
   photographs.
* `face_detection` - Find faces in a photograph or folder full for photographs.

#### `face_recognition` command line tool

The `face_recognition` command lets you recognize faces in a photograph or 
folder full  for photographs.

First, you need to provide a folder with one picture of each person you
already know. There should be one image file for each person with the
files named according to who is in the picture:

![known](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

Next, you need a second folder with the files you want to identify:

![unknown](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

Then in you simply run the command `face_recognition`, passing in
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

#### `face_detection` command line tool

The `face_detection` command lets you find the location (pixel coordinatates) 
of any faces in an image.

Just run the command `face_detection`, passing in a folder of images 
to check (or a single image):

```bash
$ face_detection  ./folder_with_pictures/

examples/image1.jpg,65,215,169,112
examples/image2.jpg,62,394,211,244
examples/image2.jpg,95,941,244,792
```

It prints one line for each face that was detected. The coordinates
reported are the top, right, bottom and left coordinates of the face (in pixels).
 
##### Adjusting Tolerance / Sensitivity

If you are getting multiple matches for the same person, it might be that
the people in your photos look very similar and a lower tolerance value
is needed to make face comparisons more strict.

You can do that with the `--tolerance` parameter. The default tolerance
value is 0.6 and lower numbers make face comparisons more strict:

```bash
$ face_recognition --tolerance 0.54 ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

If you want to see the face distance calculated for each match in order
to adjust the tolerance setting, you can use `--show-distance true`:

```bash
$ face_recognition --show-distance true ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama,0.378542298956785
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person,None
```

##### More Examples

If you simply want to know the names of the people in each photograph but don't
care about file names, you could do this:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

Barack Obama
unknown_person
```

##### Speeding up Face Recognition

Face recognition can be done in parallel if you have a computer with
multiple CPU cores. For example, if your system has 4 CPU cores, you can
process about 4 times as many images in the same amount of time by using
all your CPU cores in parallel.

If you are using Python 3.4 or newer, pass in a `--cpus <number_of_cpu_cores_to_use>` parameter:

```bash
$ face_recognition --cpus 4 ./pictures_of_people_i_know/ ./unknown_pictures/
```

You can also pass in `--cpus -1` to use all CPU cores in your system.

#### Python Module

You can import the `face_recognition` module and then easily manipulate
faces with just a couple of lines of code. It's super easy!

API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### Automatically find all the faces in an image

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
 to try it out.

You can also opt-in to a somewhat more accurate deep-learning-based face detection model.

Note: GPU acceleration (via NVidia's CUDA library) is required for good
performance with this model. You'll also want to enable CUDA support
when compliling `dlib`.

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
 to try it out.

If you have a lot of images and a GPU, you can also
[find faces in batches](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py).

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

## Python Code Examples

All the examples are available [here](https://github.com/ageitgey/face_recognition/tree/master/examples).


#### Face Detection

* [Find faces in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [Find faces in a photograph (using deep learning)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [Find faces in batches of images w/ GPU (using deep learning)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [Blur all the faces in a live video using your webcam (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

#### Facial Features

* [Identify specific facial features in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Apply (horribly ugly) digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

#### Facial Recognition

* [Find and recognize unknown faces in a photograph based on photographs of known people](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [Identify and draw boxes around each person in a photo](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [Compare faces by numeric face distance instead of only True/False matches](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [Recognize faces in live video using your webcam - Simple / Slower Version (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [Recognize faces in live video using your webcam - Faster Version (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [Recognize faces in a video file and write out new video file (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [Recognize faces on a Raspberry Pi w/ camera](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [Run a web service to recognize faces via HTTP (Requires Flask to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [Recognize faces with a K-nearest neighbors classifier](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
* [Train multiple images per person then recognize faces using a SVM](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py)

## Creating a Standalone Executable
If you want to create a standalone executable that can run without the need to install `python` or `face_recognition`, you can use [PyInstaller](https://github.com/pyinstaller/pyinstaller). However, it requires some custom configuration to work with this library. See [this issue](https://github.com/ageitgey/face_recognition/issues/357) for how to do it.

## Articles and Guides that cover `face_recognition`

- My article on how Face Recognition works: [Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
  - Covers the algorithms and how they generally work
- [Face recognition with OpenCV, Python, and deep learning](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) by Adrian Rosebrock
  - Covers how to use face recognition in practice
- [Raspberry Pi Face Recognition](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/) by Adrian Rosebrock
  - Covers how to use this on a Raspberry Pi
- [Face clustering with Python](https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/) by Adrian Rosebrock
  - Covers how to automatically cluster photos based on who appears in each photo using unsupervised learning

## How Face Recognition Works

If you want to learn how face location and recognition work instead of
depending on a black box library, [read my article](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

## Caveats

* The face recognition model is trained on adults and does not work very well on children. It tends to mix
  up children quite easy using the default comparison threshold of 0.6.
* Accuracy may vary between ethnic groups. Please see [this wiki page](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) for more details.

## <a name="deployment">Deployment to Cloud Hosts (Heroku, AWS, etc)</a>

Since `face_recognition` depends on `dlib` which is written in C++, it can be tricky to deploy an app
using it to a cloud hosting provider like Heroku or AWS.

To make things easier, there's an example Dockerfile in this repo that shows how to run an app built with
`face_recognition` in a [Docker](https://www.docker.com/) container. With that, you should be able to deploy
to any service that supports Docker images.

You can try the Docker image locally by running: `docker-compose up --build`

Linux users with a GPU (drivers >= 384.81) and [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) installed can run the example on the GPU: Open the [docker-compose.yml](docker-compose.yml) file and uncomment the `dockerfile: Dockerfile.gpu` and `runtime: nvidia` lines.

## Having problems?

If you run into problems, please read the [Common Errors](https://github.com/ageitgey/face_recognition/wiki/Common-Errors) section of the wiki before filing a github issue.

## Thanks

* Many, many thanks to [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom))
  for creating dlib and for providing the trained facial feature detection and face encoding models
  used in this library. For more information on the ResNet that powers the face encodings, check out
  his [blog post](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html).
* Thanks to everyone who works on all the awesome Python data science libraries like numpy, scipy, scikit-image,
  pillow, etc, etc that makes this kind of stuff so easy and fun in Python.
* Thanks to [Cookiecutter](https://github.com/audreyr/cookiecutter) and the
  [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template
  for making Python project packaging way more tolerable.
