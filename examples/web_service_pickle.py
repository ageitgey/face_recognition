# pylint: disable=C0321
# pylint: disable=R1705

"""
This example shows how to use face_recognition
with an image database on a webserver Flask framework.
Upload an image to see if it matches the database
Create a static folder and put your images in "images" directory
It will then generate a binary file called "db_out"
based on images in "test_image" directory
This example uses multiprocessing for faster processing
This example created based on "web_service_example.py"
"""

import os
import multiprocessing
import pickle
from pathlib import Path
import face_recognition
from flask import Flask, request, redirect, render_template, url_for
import numpy
import colorama
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMAGES_DIR = "./static/images"
DATABASE_DIR = "./static/db_out"


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid!
            # Get the filenames and pass the on to the results.html page
            return render_template(
                'results.html', results=detect_image(file)
            )

    # If no valid image file was uploaded, show the file upload form
    return render_template('homepage.html')


def db_existed(db_address):
    """
    Checks if databse file is exist
    """
    my_file = Path(db_address)
    if my_file.is_file():  # file exists
        print("database found!")
        return True
    else:
        print("database not found!")
        return False


def db_write(itemlist, db_address):
    """
    Write database file
    as "db_address"
    """
    print("writing database to file")
    with open(db_address, 'wb') as fp:
        pickle.dump(itemlist, fp)
    print("done")


def db_read(db_address):
    """
    Read the database file
    """
    with open(db_address, 'rb') as db_add:
        itemlist = pickle.load(db_add)
    return itemlist


def image_process(file):
    """
    Encode each image and return the value
    """
    size_db = 640, 480  # size of database image
    image_encoded = ([0] * 128)  # the encoded image array
    print("File name is: ", file)
    image_file = Image.open(IMAGES_DIR + '/' + file)
    image_file.thumbnail(size_db, Image.ANTIALIAS)
    image_array = numpy.array(image_file)
    # training_image = image_array
    # extract the face location
    face_locations = face_recognition.face_locations(image_array)
    # encode image array with face_locations for better result
    temp_encoding = face_recognition.face_encodings(
        image_array, face_locations)
    if temp_encoding:  # if the image encoded right it length is not zero
        image_encoded = temp_encoding[0]
        print("Encoded !")
    else:
        print("Canot encode:", file)
    return image_encoded


def detect_image(file_stream):
    """
    Recognize if an images matches the database
    """
    size_src = 640, 480  # size of uploaded image
    match_count = 0  # number of match images
    file_names = []  # list of files in images directory
    img_db_encoding = []  # list of encoding data of images
    # list of file_names that match with uploaded image
    founded_file_names = []
    # face_locations = []  # list of face_locations in each image
    im_src = Image.open(file_stream)
    # resize the image to half size
    im_src.thumbnail(size_src, Image.ANTIALIAS)
    specific_image = numpy.array(im_src)  # array of uploaded image file
    # extract the face location
    face_locations = face_recognition.face_locations(specific_image)
    # encode image array with face_locations for better result
    image_encoded = face_recognition.face_encodings(
        specific_image, face_locations)
    if image_encoded:
        specific_image_encoding = image_encoded[0]
    else:
        print("Cannot use this image!")
        redirect(url_for('upload_image'))
        return
    files = os.listdir(IMAGES_DIR)
    for file in files:
        file_names.append(file)
    print("length of file names", len(file_names))
    db_file_existed = db_existed(DATABASE_DIR)  # check if databse exist
    if db_file_existed is False:
        # Create a multiprocessing Pool based on cores of cpu
        # and generate the databse
        # you can also use pool = Poll(<a fix number>)
        # pool = Pool(os.cpu_count())
        pool = multiprocessing.Pool(2)
        # proces data_inputs iterable with pool
        img_db_encoding = pool.map(image_process, file_names)
        pool.close()
        pool.join()
    else:
        img_db_encoding = db_read(DATABASE_DIR)
    db_write(img_db_encoding, DATABASE_DIR)
    result = face_recognition.compare_faces(
        img_db_encoding, specific_image_encoding, tolerance=0.5)
    print("length of image_encoding ", len(img_db_encoding))
    for j in range(len(result)):
        if result[j]:
            match_count = match_count + 1
            founded_file_names.append(file_names[j])
    if match_count > 0:
        print("found ", match_count, " matches with names of:")
        for i in founded_file_names:
            print(i)
    else:
        print("No matches found!")
    return founded_file_names  # list of the filenames, no need for JSON


if __name__ == "__main__":
    multiprocessing.freeze_support()
    colorama.init()
    print(colorama.Back.BLUE + colorama.Fore.WHITE + colorama.Style.BRIGHT +
          "***Face Recognition System***" + colorama.Style.RESET_ALL)
    print("database directory:", os.path.abspath(IMAGES_DIR))
    print("database file:", os.path.abspath(DATABASE_DIR))
    print("*****"*20)
    print(colorama.Fore.WHITE + colorama.Style.BRIGHT)
    app.run(host='localhost', port=5001, debug=False)
