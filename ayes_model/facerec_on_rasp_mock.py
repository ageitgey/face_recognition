"""
Here we want to test the real scenario of the webcam stream on the raspberry py. 
We will also add small delays to have a complete view of the process. Further analysis will improve the model. 
"""
import os
import json
import face_recognition
import numpy as np
import cv2
import time
from mqtt_client_ayes import AyesMqttClient
import logging

log = logging.Logger("Face Recognition Logger")


def load_db(db_path):
    """
    Load the existing encodings json file, to get the existing encodings 
    """
    if os.path.exists(db_path):
        with open(db_path, 'r') as file:
            try: 
                db = json.load(file)
            except json.decoder.JSONDecodeError:
                db = []
                
    else:
        db = []
    
    # for encoding in encodings:
    #     encoding = np.array(encoding)
    
    return db

def load_test_video(test_dir, filename):
    for test_filename in os.listdir(test_dir):
        test_name, test_ext = os.path.splitext(test_filename)
        if test_name == filename:
            video_path = os.path.join(test_dir, test_filename)
    
    return cv2.VideoCapture(video_path)


def find_true_indices(boolean_list):
    """
    Find the indexes of the recognized persons 
    
    """
    return_index = -1
    for index, value in enumerate(boolean_list):
        if value:
            return_index = index 
    
    return return_index

def preprocess_frame(frame,horizontal_resizing= 0.5, vertical_resizing= 0.5):
    """
    Pre processing a frame, in particular
    1. Resize it to make it smaller, having a faster recognition process
        - The horizontal and vertical resizing values shall be (0,1]
    2. Convert from BGR (OpenCV) to RGB (used  by the face recognition model).
    Returns the small frame in rgb
    """
    
    if (horizontal_resizing > 1) or (vertical_resizing > 1):
        log.error("Accepted resizing values interval (0,1]")
        
    if (horizontal_resizing <= 0) or (vertical_resizing <= 0):
        log.error("Accepted resizing values interval (0,1]")
    
    # Resize frame of video for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=horizontal_resizing, fy=vertical_resizing)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    return rgb_small_frame


def manage_face_recognition(rgb_small_frame, face_locations, retry_next_frame, prev_faces_nb, print_logs= False):
    """
    Manages the face recognition process, updating the status of detected faces, and determining whether to retry 
    recognition in the next frame.
    
    Parameters:
    - rgb_small_frame: The current video frame in RGB format, from which faces are to be recognized.
    - face_locations: List of coordinates where faces are detected in the current frame.
    - retry_next_frame: Boolean indicating whether to retry face recognition in the next frame due to a previous unknown face.
    - prev_faces_nb: The number of faces detected in the previous frame.
    - print_logs: Boolean to indicate whether to print log messages (default is False).
    
    Returns:
    - publish_flag: Boolean indicating whether the results should be published.
    - retry_next_frame: Updated boolean indicating whether to retry face recognition in the next frame.
    - face_added_names: List of names of recognized faces.
    - prev_faces_nb: Updated number of faces detected in the current frame.
    """
    
    face_added_names = []

    
    if len(face_locations) == 0:
        
        LOGGING_STRING = "No faces are being detected"
        publish_flag = False
        
    
    elif (len(face_locations) == prev_faces_nb) and (not retry_next_frame):
        
        LOGGING_STRING = "Still here?"
        publish_flag = False
        
    
    elif (len(face_locations) != prev_faces_nb) or retry_next_frame:
        
        publish_flag = True
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        indexes = []
        if not retry_next_frame:
        
            
            # print(face_locations)
            LOGGING_STRING = "I see someone"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                index = find_true_indices(matches)
                indexes.append(index)
            # This if else condition is to give the algorithm time to recognize someone, so: 
            # the first time we see someone, if we don't recognize it we try again
            if sum(indexes) >= 0:
                LOGGING_STRING = "I know someone"
                for index in indexes:
                    face_added_names.append(names[index])
            elif -1 in indexes: # -1 is the default value we give to unkonwn encodings
                # If we don't recognize anyone, then we will redo everything the next frame we want to process! 
                LOGGING_STRING = "First Unknown Encounter"
                publish_flag = False 
                retry_next_frame = True  # Set flag to retry in the next processed frame
            else:
                log.error("You should not be here")
        
        elif retry_next_frame:
            LOGGING_STRING = "I will retry to recognize again such unknown"
            retry_next_frame = False
            publish_flag = True
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                index = find_true_indices(matches)
                indexes.append(index)
            if sum(indexes) >= 0:
                LOGGING_STRING = "I know someone"
                for index in indexes:
                    face_added_names.append(names[index])
            elif -1 in indexes: # -1 is the default value we give to unkonwn encodings
                face_added_names.append("Unknown")
                LOGGING_STRING = "Someone is not in my system"
            else:
                LOGGING_STRING = "We may have a problem here"
    
    else:
        publish_flag = False
        LOGGING_STRING = "You are in a situation not covered by the algorithm!!"
    
    if print_logs:
        print(LOGGING_STRING)
        
    return publish_flag, retry_next_frame, face_added_names, prev_faces_nb


def publish_messages(previous_names, face_added_names):
    """
    Send the messages to the MQTT broker
    The function only activates if the publish flag is set to true, in such case: 
    1. First send the names of the last recognized people on the face_removed topic
    2. Send the names of the newly recognized people on the face_added topic
    Returns the recognized faces, to be then used at step 1 the next time the function is called 
    This is needed to have a quick and clean way to avoid having the names to be repeated multiple times.
    """
    face_removed = json.dumps({"names" : previous_names})
    face_recognition_client.publish_message("greetings/face_removed", face_removed)
    face_added_names = list(set(face_added_names))
    face_added = json.dumps({"names" : face_added_names})
    face_recognition_client.publish_message("greetings/face_added", face_added)
    return face_added_names
   

 
test_dir = './test_files'  
input_movie = load_test_video(test_dir, "Video Test")
frames = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fps = input_movie.get(cv2.CAP_PROP_FPS) # get the FPS
width = int(input_movie.get(3))
height = int(input_movie.get(4))


db_path = './encodings.json'
db = load_db(db_path)
names = [encoding['name'] for encoding in db]
known_encodings = [np.array(encoding['encoding']) for encoding in db] 

print("Encodings have been loaded")
print(names)
MQTT_TOPICS = ["greetings/face_added",
               "greetings/face_removed"]
CLIENT_ID = "FaceRecognition"
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883

face_recognition_client = AyesMqttClient(
    broker= MQTT_BROKER_HOST,
    port= MQTT_BROKER_PORT,
    topics_list= MQTT_TOPICS,
    client_id= CLIENT_ID
)


# Initialize some variables
face_locations = []
face_encodings = []
sent_faces = []
frame_nb = 1
prev_faces_nb = 0
count = 0
retry_recognition = False # Flag to retry if we got a new encoding, and such encoding is unknown 
publish_flag = False # Flag to publish on mqtt
retry_next_frame = False
new_unknown_detected = False
previous_names = []

FRAMES_JUMP = 10

print("Ready to start recognition")

time.sleep(2)

face_recognition_client.connect()

while True:

    # Initialization
    count += 1 

    # Only process every other FRAMES_JUMP of video to save time
    if count % FRAMES_JUMP == 0:
        print(f"second: {count/fps}")
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!")
            break
        
        # Add the delay - based on the fps of the stream
        time.sleep(1/fps)
        
        # Pre process frame
        rgb_small_frame = preprocess_frame(frame, horizontal_resizing= 0.5, vertical_resizing= 0.5)
        
        # Find all the faces and face encodings in the current frame of video
        # face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_locations = face_recognition.face_locations(rgb_small_frame)
    
        # Main face rec AYES algo
        publish_flag, retry_next_frame, face_added_names, prev_faces_nb = manage_face_recognition(rgb_small_frame, face_locations, retry_next_frame, prev_faces_nb, print_logs= True)
    
        
        # Publish only when necessary
        if publish_flag:
            previous_names = publish_messages(previous_names, face_added_names)

            

        prev_faces_nb = len(face_locations)
    else:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!")
            break
        
    
