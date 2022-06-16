"""SVM FACE RECOGNITION WITH PROBABILITY OF UNKNOWN FACE
~ This code is written for evaluating and comparing reasons, so it isn't fully-tested.
~ Based on "face_recognition_knn.py" that uses KNN classifier.

This is an example of using the support vector machine (SVM) algorithm for face recognition.
The purpose of this method is to compare how SVM and KNN performs on the task of recognizing faces trained on a dataset
with the ability to determinate if a subject is unknown (not in the dataset).

To improve the model you should try to provide more images of each subjects, tune the thresholds and the number
of jitters.

Algorithm Description:
- In depth explanation:
https://www.youtube.com/watch?v=efR1C6CvhmE

- Fast explanation:
https://www.youtube.com/watch?v=Y6RRHw9uN9o

Usage:
1. Prepare the train dir with the known people you want to recognize:
    Train image must have only one face, otherwise it will not be used for training.
    Structure of the train dir (image format could be what is defined in the global variable "ALLOWED_EXTENSIONS"):
            <train_dir>/
            ├── <person1>/
            │   ├── <somename1>.jpeg
            │   ├── <somename2>.jpeg
            │   ├── ...
            ├── <person2>/
            │   ├── <somename1>.jpeg
            │   └── <somename2>.jpeg
            └── ...

2. Prepare the test dir with the images you want to perform predictions on.

3. Call the "train" function to produce the model. (Once you have the model you can skip this step)

4. Call the "predict" function to make predictions on an image.

NOTE:
    This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn
"""
import numpy as np

import face_recognition
from sklearn import svm
import os
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_detection_cli import image_files_in_folder


def encode_training_dataset(train_dir: str, num_jitters=0, verbose=False):
    """Encode each face in the dataset

    :param train_dir:directory that contains a sub-directory for each known person, with its name.
    :param verbose: verbosity
    :param num_jitters: (optional) How many times to jitter/resample the image.  When you set it to 100 it executes the
    face descriptor extraction 100 times on slightly modified versions of the face and returns the average result.
    You could also pick a more middle value, such as 10, which is only 10x slower, 100 is 100x slower.

    :return: List of encodings of faces and list of names in the form (encodings, names)
    """
    # X_train, y_train
    encodings = []
    names = []

    # Loop through each person in the training directory
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                face_enc = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes,
                                                           num_jitters=num_jitters)[0]
                encodings.append(face_enc)
                names.append(class_dir)

    return encodings, names


def train(train_dir: str, model_save_path=None, num_jitters=0, verbose=False):
    """Trains a support vector machine classifier for recognition on existing dataset.

    Structure of the train dir (image format could be what is defined in the global variable "ALLOWED_EXTENSIONS"):
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param train_dir: directory that contains a sub-directory for each known person, with its name.
    :param model_save_path: (optional) path to save model on disk.
    :param num_jitters: (optional) How many times to jitter/resample the image.  When you set it to 100 it executes the
    face descriptor extraction 100 times on slightly modified versions of the face and returns the average result.
    You could also pick a more middle value, such as 10, which is only 10x slower, 100 is 100x slower.
    :param verbose: verbosity of training

    :return: returns svm classifier that was trained on the given data.
    """

    # Get the 128D encodings of faces with matched names for training
    x_train, y_train = encode_training_dataset(train_dir, num_jitters, verbose)

    # Fitting classifier to the Training set
    svm_clf = svm.SVC(gamma="scale", probability=True)
    svm_clf.fit(x_train, y_train)

    # Save the trained SVM classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(svm_clf, f)

    return svm_clf


def predict(X_img_path, svm_clf=None, model_path=None, proba_threshold=0.7):
    """Recognizes faces in given image using a trained SVM classifier

    :param X_img_path: path to image to be recognized
    :param model_path: (optional) path to a pickled svm classifier. if not specified, model_save_path must be svm_clf.
    :param svm_clf: (optional) a svm classifier object. if not specified, model_save_path must be specified.
    :param proba_threshold: (optional) probability threshold for face classification. The lower it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    prediction = []

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if svm_clf is None and model_path is None:
        raise Exception("Must supply svm classifier either thourgh svm_clf or model_path")

    # Load a trained svm model (if one was passed in)
    if svm_clf is None:
        with open(model_path, 'rb') as f:
            svm_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations, num_jitters=0)

    # Use the SVM model to find the best matches for the test face
    faces_probabilities = svm_clf.predict_proba(faces_encodings)

    # For each prediction find if is an unknown
    for i in range(len(faces_encodings)):
        name = "0"

        # Find the index of the "most close" prediction
        max_index = np.argmax(faces_probabilities[i])

        # Check id the "most close" prediction is greater than the probability threshold
        if faces_probabilities[i][max_index] >= proba_threshold:
            name = max_index + 1

        prediction.append((str(name), X_face_locations[i]))

    return prediction


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        name = NAMES[int(name)]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


def load_names_from_dataset(train_dir, unknown_label="unknown"):
    """Load name of the people in the dataset based on the name of each folder.

    :param train_dir: path to the train dir, for getting the names.
    :param unknown_label: (optional) label used to mark unknown people.
    :return: list of names.
    """
    names = [unknown_label]
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        names.append(class_dir)

    return names


# Config vars
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

TRAIN_DIR = "people_dataset/train"
TEST_DIR = "people_dataset/test"
SVM_MODEL_NAME = "trained_svm_model.clf"

NAMES = load_names_from_dataset(TRAIN_DIR)


if __name__ == "__main__":
    # STEP 1: Train the SVM classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training SVM classifier...")
    classifier = train(TRAIN_DIR, model_save_path=SVM_MODEL_NAME, num_jitters=1)
    print("Training complete!")
    print(f"[INFO] People labels: {NAMES}")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir(TEST_DIR):
        full_file_path = os.path.join(TEST_DIR, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=SVM_MODEL_NAME, proba_threshold=0.4)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(NAMES[int(name)], left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join(TEST_DIR, image_file), predictions)
