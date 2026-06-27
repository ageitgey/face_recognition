import common
import os
import cv2



if __name__ == "__main__":
    TRAIN_FOLDER = './train_files/'
    ENCODINGS_PATH = './encodings.json'
    
    user_name = input("New encoding:")
    
    try :
        os.mkdir(TRAIN_FOLDER + user_name)
    except FileExistsError:
        print("User name already exists")
        exit()
    
    NUMBER_OF_PICTURES = 10
    
    for i in range(1, NUMBER_OF_PICTURES + 1):        
        input("Press enter to take picture (" + str(i) + "/" + str(NUMBER_OF_PICTURES) + ")")
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened(): break
        ret, frame = camera.read()
        camera.release()
        if not ret: break
        
        cv2.imwrite(TRAIN_FOLDER + user_name + '/' + user_name + str(i) + ".png", frame)
    
    
    new_encodings = common.process_encoding_folder(TRAIN_FOLDER + user_name, user_name)
    common.append_encodings(ENCODINGS_PATH, new_encodings)
    