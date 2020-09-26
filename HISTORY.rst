History
=======

1.4.0 (2020-09-26)
------------------

* Dropping support for Python 2.x
* --upsample a parameter for command line face_recognition

1.3.0 (2020-02-20)
------------------

* Drop support for Python 3.4 and add 3.8
* Blink detection example


1.2.3 (2018-08-21)
------------------

* You can now pass model="small" to face_landmarks() to use the 5-point face model instead of the 68-point model.
* Now officially supporting Python 3.7
* New example of using this library in a Jupyter Notebook

1.2.2 (2018-04-02)
------------------

* Added the face_detection CLI command
* Removed dependencies on scipy to make installation easier
* Cleaned up KNN example and fixed a bug with drawing fonts to label detected faces in the demo


1.2.1 (2018-02-01)
------------------

* Fixed version numbering inside of module code.


1.2.0 (2018-02-01)
------------------

* Fixed a bug where batch size parameter didn't work correctly when doing batch face detections on GPU.
* Updated OpenCV examples to do proper BGR -> RGB conversion
* Updated webcam examples to avoid common mistakes and reduce support questions
* Added a KNN classification example
* Added an example of automatically blurring faces in images or videos
* Updated Dockerfile example to use dlib v19.9 which removes the boost dependency.


1.1.0 (2017-09-23)
------------------

* Will use dlib's 5-point face pose estimator when possible for speed (instead of 68-point face pose esimator)
* dlib v19.7 is now the minimum required version
* face_recognition_models v0.3.0 is now the minimum required version


1.0.0 (2017-08-29)
------------------

* Added support for dlib's CNN face detection model via model="cnn" parameter on face detecion call
* Added support for GPU batched face detections using dlib's CNN face detector model
* Added find_faces_in_picture_cnn.py to examples
* Added find_faces_in_batches.py to examples
* Added face_rec_from_video_file.py to examples
* dlib v19.5 is now the minimum required version
* face_recognition_models v0.2.0 is now the minimum required version


0.2.2 (2017-07-07)
------------------

* Added --show-distance to cli
* Fixed a bug where --tolerance was ignored in cli if testing a single image
* Added benchmark.py to examples


0.2.1 (2017-07-03)
------------------

* Added --tolerance to cli


0.2.0 (2017-06-03)
------------------

* The CLI can now take advantage of multiple CPUs. Just pass in the -cpus X parameter where X is the number of CPUs to use.
* Added face_distance.py example
* Improved CLI tests to actually test the CLI functionality
* Updated facerec_on_raspberry_pi.py to capture in rgb (not bgr) format.


0.1.14 (2017-04-22)
-------------------

* Fixed a ValueError crash when using the CLI on Python 2.7


0.1.13 (2017-04-20)
-------------------

* Raspberry Pi support.


0.1.12 (2017-04-13)
-------------------

* Fixed: Face landmarks wasn't returning all chin points.


0.1.11 (2017-03-30)
-------------------

* Fixed a minor bug in the command-line interface.


0.1.10 (2017-03-21)
-------------------

* Minor pref improvements with face comparisons.
* Test updates.


0.1.9 (2017-03-16)
------------------

* Fix minimum scipy version required.


0.1.8 (2017-03-16)
------------------

* Fix missing Pillow dependency.


0.1.7 (2017-03-13)
------------------

* First working release.
