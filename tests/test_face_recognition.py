#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_face_recognition
----------------------------------

Tests for `face_recognition` module.
"""


import unittest
import os
from click.testing import CliRunner

from face_recognition import api
from face_recognition import cli


class Test_face_recognition(unittest.TestCase):

    def test_load_image_file(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        self.assertEqual(img.shape, (1137, 910, 3))

    def test_load_image_file_32bit(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', '32bit.png'))
        self.assertEqual(img.shape, (1200, 626, 3))

    def test_raw_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        detected_faces = api._raw_face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0].top(), 142)
        self.assertEqual(detected_faces[0].bottom(), 409)

    def test_raw_face_locations_32bit_image(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', '32bit.png'))
        detected_faces = api._raw_face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0].top(), 290)
        self.assertEqual(detected_faces[0].bottom(), 558)

    def test_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (142, 617, 409, 349))

    def test_partial_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama_partial_face.jpg'))
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (142, 191, 365, 0))

        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama_partial_face2.jpg'))
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (142, 551, 409, 349))

    def test_raw_face_landmarks(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        face_landmarks = api._raw_face_landmarks(img)
        example_landmark = face_landmarks[0].parts()[10]

        self.assertEqual(len(face_landmarks), 1)
        self.assertEqual(face_landmarks[0].num_parts, 68)
        self.assertEqual((example_landmark.x, example_landmark.y), (552, 399))

    def test_face_landmarks(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        face_landmarks = api.face_landmarks(img)

        self.assertEqual(
            set(face_landmarks[0].keys()),
            set(['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
                 'nose_tip', 'left_eye', 'right_eye', 'top_lip',
                 'bottom_lip']))
        self.assertEqual(
            face_landmarks[0]['chin'],
            [(369, 220), (372, 254), (378, 289), (384, 322), (395, 353),
             (414, 382), (437, 407), (464, 424), (495, 428), (527, 420),
             (552, 399), (576, 372), (594, 344), (604, 314)])

    def test_face_encodings(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        encodings = api.face_encodings(img)

        self.assertEqual(len(encodings), 1)
        self.assertEqual(len(encodings[0]), 128)

    def test_compare_faces(self):
        img_a1 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        img_a2 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama2.jpg'))
        img_a3 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama3.jpg'))

        img_b1 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'biden.jpg'))

        face_encoding_a1 = api.face_encodings(img_a1)[0]
        face_encoding_a2 = api.face_encodings(img_a2)[0]
        face_encoding_a3 = api.face_encodings(img_a3)[0]
        face_encoding_b1 = api.face_encodings(img_b1)[0]

        faces_to_compare = [
            face_encoding_a2,
            face_encoding_a3,
            face_encoding_b1]

        match_results = api.compare_faces(faces_to_compare, face_encoding_a1)
        self.assertTrue(match_results[0])
        self.assertTrue(match_results[1])
        self.assertFalse(match_results[2])

    def test_command_line_interface(self):
        target_string = '--help  Show this message and exit.'
        runner = CliRunner()
        help_result = runner.invoke(cli.main, ['--help'])
        self.assertEqual(help_result.exit_code, 0)
        self.assertTrue(target_string in help_result.output)
