import face_recognition
import cv2
import numpy as np
from flask import Flask, jsonify, request, redirect, make_response
from flask_restplus import Api, Namespace, Resource, reqparse
from werkzeug.utils import cached_property
from werkzeug.datastructures import FileStorage
import logging, sys, time



logging.basicConfig(stream=sys.stderr, level=logging.INFO)
app = Flask(__name__)
api = Api()
api.init_app(app)



upload_ns = Namespace('upload')
api.add_namespace(upload_ns)

image_parser = upload_ns.parser()
image_parser.add_argument('image',type=FileStorage,location='files',required=True,help='Image of a person')
video_capture = cv2.VideoCapture(0)

ALLOWED_EXTENSIONS = {'image/png', 'image/jpg', 'image/jpeg', 'image/gif'}


@upload_ns.route('/')
class upload(Resource):
    @api.expect(image_parser,validate=True)       
    def post(self):
        args =  image_parser.parse_args()
        if args['image'] and args['image'].mimetype in ALLOWED_EXTENSIONS:
            found = detect_faces(args['image'].stream)
            if found:
                return make_response(jsonify("Msg: Person Found"),200)
            else:
                return make_response(jsonify("Msg: Person not found"),417)
        else: 
            return make_response(jsonify("Msg: Upload an image"),415)



def detect_faces(file_stream):

    start_time = time.perf_counter()

    try: 
        person_tobe_found = face_recognition.load_image_file(file_stream)
        person_tbf_encoding = face_recognition.face_encodings(person_tobe_found)[0]

        known_face_encodings = [
            person_tbf_encoding
        ]
        known_face_names = [
            "Person"
        ]

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
    except: 
        return make_response(jsonify("Msg: No face found"))


    while True:
        
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame,None, fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance= 0.6)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]                   
                
                face_names.append(name)
                logging.debug(face_names)
                if 'Person' in face_names:
                    return True

        process_this_frame = not process_this_frame


        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        end_time = time.perf_counter()
        run_time = end_time - start_time

        """cv2.imshow('Video', frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """
        
        if run_time > 15: # No. of seconds we look for that person 
            return False

    video_capture.release()
    cv2.destroyAllWindows()


