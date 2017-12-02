#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
from os.path import join, isfile


from examples import face_recognition_knn


class Test_face_recognition(unittest.TestCase):
    def test_knn1(self):
        knn_clf = face_recognition_knn.train(join(os.path.dirname(__file__), "test_images_knn/train"))

        assert("obama" in face_recognition_knn.predict(join(os.path.dirname(__file__), "test_images_knn/test", "obama1.jpg"), knn_clf=knn_clf)[0])

        assert("alex_lacamoire" in face_recognition_knn.predict(join(os.path.dirname(__file__), "test_images_knn/test", "alex_lacamoire1.jpg"), knn_clf=knn_clf)[0])

        assert("kit_harington" in face_recognition_knn.predict(join(os.path.dirname(__file__), "test_images_knn/test", "johnsnow_test1.jpg"), knn_clf=knn_clf)[0])

        assert(sorted(["kit_harington", "rose_leslie"]) == sorted([pred[0] for pred in
                                                                   face_recognition_knn.predict(
                                                                       join(os.path.dirname(__file__), "test_images_knn/test", "kit_with_rose.jpg"),
                                                                       knn_clf=knn_clf)]))

        assert(sorted(["obama", "biden", "N/A"]) == sorted([pred[0] for pred in
                                                            face_recognition_knn.predict(
                                                                join(os.path.dirname(__file__), "test_images_knn/test", "obama_and_biden.jpg"),
                                                                knn_clf=knn_clf)]))

    def test_knn_pickle(self):
        pickle_path = join(os.path.dirname(__file__), "test_images_knn/knn_clf.p")
        face_recognition_knn.train(join(os.path.dirname(__file__), "test_images_knn/train"), model_save_path=pickle_path)

        try:
            assert("obama" in face_recognition_knn.predict(join(os.path.dirname(__file__), "test_images_knn/test", "obama1.jpg"), model_save_path=pickle_path)[0])

            assert("alex_lacamoire" in face_recognition_knn.predict(join(os.path.dirname(__file__), "test_images_knn/test", "alex_lacamoire1.jpg"), model_save_path=pickle_path)[0])

            assert("kit_harington" in face_recognition_knn.predict(join(os.path.dirname(__file__), "test_images_knn/test", "johnsnow_test1.jpg"), model_save_path=pickle_path)[0])

            assert(sorted(["kit_harington", "rose_leslie"]) == sorted([pred[0] for pred in face_recognition_knn.predict(
                join(os.path.dirname(__file__), "test_images_knn/test", "kit_with_rose.jpg"),
                model_save_path=pickle_path)]))

            assert(sorted(["obama", "biden", "N/A"]) == sorted([pred[0] for pred in
                                                                face_recognition_knn.predict(
                                                                    join(os.path.dirname(__file__), "test_images_knn/test", "obama_and_biden.jpg"),
                                                                    model_save_path=pickle_path)]))
        finally:
            if isfile(pickle_path):
                os.remove(pickle_path)
