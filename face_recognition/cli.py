# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import scipy.misc
import warnings
import face_recognition.api as face_recognition


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


def test_image(image_to_check, known_names, known_face_encodings):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if unknown_image.shape[1] > 1600:
        scale_factor = 1600 / unknown_image.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unknown_image = scipy.misc.imresize(unknown_image, scale_factor)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

        if True in result:
            [print("{},{}".format(image_to_check, name)) for is_match, name in zip(result, known_names) if is_match]
        else:
            print("{},unknown_person".format(image_to_check))


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


@click.command()
@click.argument('known_people_folder')
@click.argument('image_to_check')
def main(known_people_folder, image_to_check):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    if os.path.isdir(image_to_check):
        [test_image(image_file, known_names, known_face_encodings) for image_file in image_files_in_folder(image_to_check)]
    else:
        test_image(image_to_check, known_names, known_face_encodings)


if __name__ == "__main__":
    main()
