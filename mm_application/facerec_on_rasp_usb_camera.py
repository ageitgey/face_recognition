import face_recognition
import numpy as np
import cv2
import os
import json
import time
from mqtt_client_ayes import AyesMqttClient


# KEPT LIKE THIS AS THIS MAY MOVE AROUND IN THE FUTURE
ENCODINGS_PATH = "./encodings.json"

if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, 'r') as file:
        try: 
            db = json.load(file)
        except json.decoder.JSONDecodeError:
            db = []

names = [encoding['name'] for encoding in db]
known_encodings = [np.array(encoding['encoding']) for encoding in db]

mqtt_handler_client = AyesMqttClient(
    broker = "localhost",
    port = 1883,
    topics_list = ["greetings/face_added", "greetings/face_removed"],
    client_id = "FaceRecognition"
)

# Wait until mqtt handler is running
while True:
    try :
        mqtt_handler_client.connect()
        break

    except Exception as e:
        print(e)
        time.sleep(1)


def find_true_indices(boolean_list):
    """
    Find the indexes of the recognized persons 
    
    """
    return_index = -1
    for index, value in enumerate(boolean_list):
        if value:
            return_index = index 
    
    return return_index


def preprocess_frame(frame):
    """
    Pre processing a frame, in particular
    1. Resize it to make it smaller, having a faster recognition process
        - The horizontal and vertical resizing values shall be [0,1]
    Returns the small frame in rgb
    """
    
    RESIZING = 0.5
    
    # Resize frame of video for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZING, fy=RESIZING)

    return small_frame


def detect_faces(rgb_small_frame):
    """
    Manages the face recognition process from current video frame in RGB and returns the recognized people on screen.
    """
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if len(face_locations) == 0: return []
    
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    people_on_screen = []
    
    # See if the face is a match for the known face(s)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        index = find_true_indices(matches)
        
        if index == -1:
            continue # People is unknown. How do we handle that ?
        
        detected_people = names[index]
        if detected_people not in people_on_screen:
            people_on_screen.append(detected_people)
            
    return people_on_screen


def publish_messages(removed_people, added_people):
    """
    Send the messages to the MQTT broker
    The function only activates if the publish flag is set to true, in such case: 
    1. First send the names of the last recognized people on the face_removed topic
    2. Send the names of the newly recognized people on the face_added topic
    Returns the recognized faces, to be then used at step 1 the next time the function is called 
    This is needed to have a quick and clean way to avoid having the names to be repeated multiple times.
    """
    face_removed = json.dumps({"names" : removed_people})
    mqtt_handler_client.publish_message("greetings/face_removed", face_removed)
    face_added = json.dumps({"names" : added_people})
    mqtt_handler_client.publish_message("greetings/face_added", face_added)


if __name__ == "__main__":    
    cap = cv2.VideoCapture(0)
        
    if not cap.isOpened(): exit

    frame_i = 0
    previous_people_on_frame = []
    
    people_list = {}
    for name in names:
        if name not in people_list:
            people_list[name] = {
                "last_time_became_visible": 0,
                "last_time_stayed_visible": 0,
                "is_visible" : False
            }
    
    while True:
        ret, frame = cap.read()
        
        must_process_frame = ((frame_i % 2) == 0) and ret
        frame_i += 1
        
        if must_process_frame:            
            current_time = time.time()
            
            rgb_small_frame = preprocess_frame(frame) 
            current_people_on_frame = detect_faces(rgb_small_frame)

            publish_added_people = []
            publish_removed_people = []

            for people in people_list:
                # Short-term
                
                if (people not in previous_people_on_frame) and (people in current_people_on_frame):
                    # people becomes (short-term) visible
                    people_list[people]["last_time_became_visible"] = current_time
                
                if (people in previous_people_on_frame) and (people in current_people_on_frame):
                    # people stays (short-term) visible
                    people_list[people]["last_time_stayed_visible"] = current_time
                    
                # Long-term
                VISIBLE_THRESHOLD_DURATION = 0.5
                NOT_VISIBLE_THRESHOLD_DURATION = 5
                
                if not people_list[people]["is_visible"]:
                    if (people_list[people]["last_time_stayed_visible"] - people_list[people]["last_time_became_visible"]) >= VISIBLE_THRESHOLD_DURATION:
                        # people becomes (long-term) visible
                        people_list[people]["is_visible"] = True
                        publish_added_people.append(people)
                        
                else:
                    if (current_time - people_list[people]["last_time_stayed_visible"]) >= NOT_VISIBLE_THRESHOLD_DURATION:
                        # people becomes (long-term) invisible
                        people_list[people]["is_visible"] = False
                        people_list[people]["last_time_became_visible"] = 0
                        people_list[people]["last_time_stayed_visible"] = 0
                        publish_removed_people.append(people)

            previous_people_on_frame = current_people_on_frame
            
            if (len(publish_added_people) > 0) or (len(publish_removed_people) > 0):
                publish_messages(publish_removed_people, publish_added_people)
                print(publish_removed_people, publish_added_people)
            
        time.sleep(0.01)
