import os
import json
import face_recognition



def save_encodings(encodings_path, encodings):
    """
    Save the new encodings in the encodings.json file
    """
    with open(encodings_path, 'w') as file:
        json.dump(encodings, file, indent=4)

def load_encodings(encodings_path):
    """
    Load the existing encodings json file, to get the existing encodings 
    """
    if os.path.exists(encodings_path):
        with open(encodings_path, 'r') as file:
            try: 
                return json.load(file)
            except json.decoder.JSONDecodeError as e:
                print(e)
    
    return []

def append_encodings(encodings_path, new_encodings):
    """
    Load the existing encodings json file, and append the given list of new encodings
    """
    old_encodings = load_encodings(encodings_path)
    encodings = old_encodings + new_encodings
    save_encodings(encodings_path, encodings)

def create_encoding(image_path):
    """
    Use the face recognition api to load the image and create the new encoding
    """
    picture = face_recognition.load_image_file(file=image_path)
    face_bounding_boxes = face_recognition.face_locations(picture)

    if len(face_bounding_boxes) == 1:
        return face_recognition.face_encodings(picture, num_jitters=50, model="large")[0]
    else:
        print("Number of detected faces is not 1 in ", image_path, "(", len(face_bounding_boxes), ")")
        return None

def process_encoding_folder(encoding_folder, name):
    """
    Process the images in the given folder and return a list of the encodings
    """
    new_encodings = []
    
    for file_name in os.listdir(encoding_folder):
        image_path = os.path.join(encoding_folder, file_name)
        new_encoding = create_encoding(image_path)
        if new_encoding is not None:
            print(f"New encoding created " + name)
            new_encodings.append({'name': name, 'encoding': new_encoding.tolist()})
    
    return new_encodings
