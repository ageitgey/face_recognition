# AYES Face Recognition Project 
The first application of the AYES-MagicMirror project is based on computer vision. 
The expected output is to have a face recognition algorithm to recognize the people
of the office (obviously only the ones that accepted to have their encodings taken).

To start, we decided to create a fork from this [face recognition project](https://github.com/ageitgey/face_recognition), 
in the future we will move towards a fully in-house application. This means that the model is based on the [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
Labeled Faces in the Wild benchmark (see the [LFW paper (PDF)](https://people.cs.umass.edu/~elm/papers/lfw.pdf)).

If the following README file is missing something you're looking for, the first could be referring to the [original project README](https://github.com/ageitgey/face_recognition)! (This file, like all the rest of the application, is built on that fundamentals.)


## Prerequisites 
- Docker Desktop (ADD LINK TO THE INTERNAL DOCUMENTATION ABOUT IT)
- x11 server forwarding (ADD LINK TO THE INTERNAL DOCUMENTATION) - only if in development you *really want* semi-live screen forwarding of the result of whatever you are developing. All the necessary dependencies will be commented out in both the dockerfile and docker-compose development files, if *really want* it uncomment the section. 

**CLONE REPOSITORY**
As always , go into the folder where you want to create the project, open the terminal, and type the following command
```
git clone https://github.com/AndreaCalabro-AYES/face_recognition.git
```
## Project structure
To enable a lean development of the MagicMirror project, we decided to opt for fully containerized applications. As you may know, the target of the MagicMirror is a Raspberry Pi, and containers go as far as the Host HW (REFRESH LINK). The workaround, that allowed also to opt for lighter containers on target, has been to create different images and work environments
- Development, intended for Win/Mac applications 
  * The Container
    + The *Dockerfile.dev* contains the dependencies to run the face recognition and eventually the x11 server on Windows machines 
    + The *docker-compose.dev.yml* runs the relative dockerfile, connects to the custom Ayes-MM bridge network, sets the logging not to be buffered (quite useful during the dev phase of the face recognition), and creates a specifc volume allowing to work on the algorithm without needing to re-build each time. Why no TTY? We'll see if in the future we'll create different branches for dev and target. As the name is not the conventional one, you will need to run it in a different way (look below)
  * The work environment
    + The volume is based on the *ayes_model* sub directory, so you should work in there
    + You will be able to perform the actions described here
- GPU, intended for GPU-provided host machines - it builds and runs, no more work performed but kept as you never know
  * The Container
    + Image Content
    + Docker-compose
  * The work environment
- Target, lightest image designed to be integrated on target within the full MagicMirror Project
  * The Container
    + The *Dockerfile.rasp* contains the dependencies to run the face recognition on the Raspberry Pi (ARM32 CPU)
    + The *docker-compose.yml* RE-WORKS NEEDED --> no network, incorrect volume (do we even need one?)
  * The work environment
    + The folder is *mm_application*, this contains a json db and two python scripts
    + Find here what is perfomed in there

 
### Target (Raspberry Pi) Work Environment
The *mm_application* folder contains the following files:
1. facerec_on_rasp_usb_camera.py, the python script where we implement the final application algorithm
  - Read the camera input
  - Perform face recognition, using the algorithm implemented in the development work environment
  - Send the information via our MQTT communication - find here its [README](README.md) file
2. facerec_on_rasp.py, the python script that shall be used in case of pycamera module instead of usb webcam (still untouched)
3. Encodings json file where we store "name": 128-np.array of encodings generated running the correct algorithm. This process
is yet to be defined, soon we will migrate everything to AWS so tune back for more info. 

### Development Work Environment
The *ayes_model* folder contains all the files that are needed to perform two main actions:
1. **Tune the model**
Done with 
In the ayes_model folder there are two subfolders
- test_files
- train_files

In both you will need to put a different picture of you, the one in train_files will be used to tune the model and the one in test_files to test the encoding ! If the test will be passed, the encoding will be created in the encodings db. In this way, you can really work on your local machine to create your encoding, and this is probably the most data-wise option we have!  

> [!IMPORTANT]
> Give the correct name to both: the name of the tuning image will be used as identifier for your face encoding from now on, and to look for the correct test file!

In the docker-compose file launch the command
```
python3 create_face_encoding_db
```
This will automatically manage the update of a database (now json, need optimization here as well).
After this you will have to delete both pictures from the folder - once we will move to AWS we will also manage the automatic deletion of the pictures.

2. **Develop the algorithm**
This can be done with screen forwarding, or without. Here we will only explain the latter case, as the former solution is a mix between this and what is explained here (LINK TO X11 SCREEN FORWARDING)

The algorithm is implemented in a script that mocks the raspberry pi use case. 
The setup is
- A video, called "Video Test", shall be used as input. Not using live recording makes things
  - faster
  - repeatable
  - simpler
- An MQTT client, more info [here](https://github.com/AndreaCalabro-AYES/MQTT_Broker/blob/development/README.md#client)
- The algorithm that looks for faces in a frame, and recognizes any encodings corresponding to one in its database (the encodings.js). 

## Next Steps
Many improvements are on the timeline 
1. Optimization of this heavy task
2. Migrate towards a self developed ML model, reducing the amount of dependencies

