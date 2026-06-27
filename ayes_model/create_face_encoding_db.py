"""
This file is intended to create the database used to recognize the faces for the ayes application
"""

import common
import os



if __name__ == "__main__":
    ENCODINGS_PATH = './encodings.json'
    TRAIN_FOLDER = './train_files/'
    
    new_encodings = []
    
    for sub_dir in os.listdir(TRAIN_FOLDER):        
        new_encodings = new_encodings + common.process_encoding_folder(TRAIN_FOLDER + sub_dir, sub_dir)
        
    common.save_encodings(ENCODINGS_PATH, new_encodings)
