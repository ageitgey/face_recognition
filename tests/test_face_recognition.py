#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_face_recognition
----------------------------------

Tests for `face_recognition` module.
"""


import unittest
import os
import numpy as np
from click.testing import CliRunner

from face_recognition import api
from face_recognition import face_recognition_cli
from face_recognition import face_detection_cli


class Test_face_recognition(unittest.TestCase):

    def test_load_image_url(self):
        img = api.load_image_url("https://homepages.cae.wisc.edu/~ece533/images/cat.png")
        self.assertLessEqual(img.shape, (1137, 910, 3))

    
    
    
    def test_face_distant(self):
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

        distant_results = api.face_distant(faces_to_compare, face_encoding_a1)

        # 0.6 is the default face distance match threshold. So we'll spot-check that the numbers returned
        # are above or below that based on if they should match (since the exact numbers could vary).
        self.assertEqual(type(distant_results), np.ndarray)
        self.assertLessEqual(distant_results[0], 0.6)
        self.assertLessEqual(distant_results[1], 0.6)
        self.assertLessEqual(distant_results[2], 0.6)

    def test_face_distant_empty_lists(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'biden.jpg'))
        face_encoding = api.face_encodings(img)[0]

        # empty python list
        faces_to_compare = []

        distant_results = api.face_distant(faces_to_compare, face_encoding)
        self.assertEqual(type(distant_results), np.ndarray)
        self.assertEqual(len(distant_results), 0)

        # empty numpy list
        faces_to_compare = np.array([])

        distant_results = api.face_distant(faces_to_compare, face_encoding)
        self.assertEqual(type(distant_results), np.ndarray)
        self.assertEqual(len(distant_results), 0)

  

    
    
    

    