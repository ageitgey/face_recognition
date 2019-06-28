import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

import face_recognition

from flask import jsonify
from flask import Flask
from flask import request
import traceback

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect():

    input_path = generate_random_filename(upload_directory,"jpg")

    try:
        url = request.json["url"]

        download(url, input_path)
       
        results = []

        image = face_recognition.load_image_file(input_path)
        locations = face_recognition.face_locations(image)

        for location in locations:
            results.append({
                'y-top': location[0],
                'x-top': location[1],
                'y-bottom': location[2],
                'x-bottom': location[3]
                })

        return json.dumps(results), 200


    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


    finally:
        clean_all([
            input_path
        ])


if __name__ == '__main__':
    global upload_directory
    global model, graph
    global img_width, img_height
    global class_names
    

    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)

