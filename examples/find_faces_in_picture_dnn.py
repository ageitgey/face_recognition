#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dlib
import cv2
from PIL import Image
import os
import face_recognition
def detectFace(net,img,model_path):
    """
    Returns an array of bounding boxes of human faces in a image
    :param net:the network
    :param img:the image
    :return:A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    conf_threshold = 0.5

    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, img,conf_threshold)
    face_locations_list = []
    for box in bboxes:
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        face_locations = _trim_css_to_bounds(_rect_to_css(rect), img.shape)
        face_locations_list.append(face_locations)
    # print("list0", face_locations_list)
    return face_locations_list

def detectFaceOpenCVDnn(net, frame,conf_threshold):
    """
    the image is converted to a blob and passed through the network using the forward() function.
    The output detections is a 4-D matrix, where
    The 3rd dimension iterates over the detected faces. (i is the iterator over the number of faces)
    The fourth dimension contains information about the bounding box and score for each face.
    For example, detections[0,0,0,2] gives the confidence score for the first face, and detections[0,0,0,3:6] give the bounding box
    :param net:the network
    :param frame:the image
    :param conf_threshold:the confidence score
    :return: image with box
    """
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
#     print("detections---",detections)
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            # print("(x1, y1), (x2, y2)",(x1, y1), (x2, y2))
#             bboxes.append([409L, 564L, 761L, 212L])
#             cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes
def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()
if __name__ == "__main__":
    model_path=os.path.join("dnn_model/")
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("biden.jpg")
    DNN = "TF"
    #     DNN = "CAFFE"
    if DNN == "CAFFE":
        modelFile = model_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = model_path + "deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = model_path + "opencv_face_detector_uint8.pb"
        configFile = model_path + "opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    # Find all the faces in the image using the default DNN-based model.
    # This method is fairly accurate, and it's more accurated than the CNN model and it can run GPU accelerated.
    face_locations = detectFace(net,image, model_path=model_path)
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()

