from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree'):
    X = []
    y = []
    name_to_num_samples = {}
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        num_samples = 0
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            #face_bbox = faces_bboxes[0]
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)
            num_samples += 1
        name_to_num_samples[class_dir] = num_samples

    avg_samples_per_name = int(sum(name_to_num_samples.values()) / float(len(name_to_num_samples.keys())))

    if n_neighbors is None:
        n_neighbors = int(max(avg_samples_per_name/2, 1))
        print("Chose n_neighbors as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo)
    knn_clf.fit(X, y)

    if model_save_path != "":
        pickle.dump(knn_clf, open(model_save_path, 'wb'))
    return knn_clf

def predict(X_img_path, knn_clf = None, model_save_path ="", DIST_THRESH = .5):
    """
    returns [(name, bounding box), ...]
    :param X_img_path:
    :param knn_clf:
    :param model_save_path:
    :param thresh:
    :return:
    """

    if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if knn_clf is None:
        knn_clf = pickle.load(open(model_save_path, 'rb'))

    X_img = face_recognition.load_image_file(X_img_path)
    X_faces_loc = face_locations(X_img)
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)


    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [True if closest_distances[0][i][0] <= DIST_THRESH else False for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]

def draw_preds(img_path, preds):
    source_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for pred in preds:
        loc = pred[1]
        name = pred[0]
        # (top, right, bottom, left) => (left,top,right,bottom)
        draw.rectangle(((loc[3], loc[0]), (loc[1],loc[2])), outline="red")
        draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
    source_img.show()

if __name__ == "__main__":
    #knn_clf = None
    knn_clf = train("knn_examples/train")
    for img_path in listdir("knn_examples/test"):
        preds = predict(join("knn_examples/test", img_path) ,knn_clf=knn_clf)
        print(preds)
        draw_preds(join("knn_examples/test", img_path), preds)
