import face_recognition
import cv2
import numpy as np
import def_load as load
import random as rn

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
print("GO!")
for a in range(10):
    print("trying", a)
    video_capture = cv2.VideoCapture(a, cv2.CAP_DSHOW)
    ret, frame = video_capture.read()
    print(ret)
    if ret == True:
        break
    else:
        video_capture.release()

print("video captured")
# print(video_capture)
picture_num = 0
# Load a sample picture and learn how to recognize it.

AbrahamovaTv = load.img("faces/ucitele/AbrahamovaTv.jpg")
Adamikova2AjRuD = load.img("faces/ucitele/Adamikova2AjRuD.jpg")
BednarikovaMaITV = load.img("faces/ucitele/BednarikovaMaITV.jpg")
BerankovaAjFr = load.img("faces/ucitele/BerankovaAjFr.jpg")
BraunerChBi = load.img("faces/ucitele/BraunerChBi.jpg")
CastellarSp = load.img("faces/ucitele/CastellarSp.jpg")
CechovaAj = load.img("faces/ucitele/CechovaAj.jpg")
ChromaTvZ = load.img("faces/ucitele/ChromaTvZ.jpg")
DanielTvZ = load.img("faces/ucitele/DanielTvZ.jpg")
DohnalovaCjHv = load.img("faces/ucitele/DohnalovaCjHv.jpg")
DostalovaTv = load.img("faces/ucitele/DostalovaTv.jpg")
DubovaNeCj = load.img("faces/ucitele/DubovaNeCj.jpg")
DvorskyAjD = load.img("faces/ucitele/DvorskyAjD.jpg")
FaberovaAjRu = load.img("faces/ucitele/FaberovaAjRu.jpg")
FerencMaFy = load.img("faces/ucitele/FerencMaFy.jpg")
FukaCjNe = load.img("faces/ucitele/FukaCjNe.jpg")
GlosCjD = load.img("faces/ucitele/GlosCjD.jpg")
gosD = load.img("faces/ucitele/gosD.jpg")
HajduMaAj = load.img("faces/ucitele/HajduMaAj.jpg")
HajekBiCh = load.img("faces/ucitele/HajekBiCh.jpg")
HamrikovaMaCh = load.img("faces/ucitele/HamrikovaMaCh.jpg")
HavranovaAj = load.img("faces/ucitele/HavranovaAj.jpg")
HermanovaCjHvRu = load.img("faces/ucitele/HermanovaCjHvRu.jpg")
HorvathSp = load.img("faces/ucitele/HorvathSp.jpg")
HubackovaCjD = load.img("faces/ucitele/HubackovaCjD.jpg")
JanickovaAjCj = load.img("faces/ucitele/JanickovaAjCj.jpg")
JonesAj = load.img("faces/ucitele/JonesAj.jpg")
JurikTvZ = load.img("faces/ucitele/JurikTvZ.jpg")
KalandrovaCjOv = load.img("faces/ucitele/KalandrovaCjOv.jpg")
KnapekAjSp = load.img("faces/ucitele/KnapekAjSp.jpg")
KonecnaNeCj = load.img("faces/ucitele/KonecnaNeCj.jpg")
KostkovaMaCh = load.img("faces/ucitele/KostkovaMaCh.jpg")
KovarovaMaCh = load.img("faces/ucitele/KovarovaMaCh.jpg")
KralovaAj = load.img("faces/ucitele/KralovaAj.jpg")
KrcovaMaZ = load.img("faces/ucitele/KrcovaMaZ.jpg")
KrejcirovaSp = load.img("faces/ucitele/KrejcirovaSp.jpg")
KrejcirTvZ = load.img("faces/ucitele/KrejcirTvZ.jpg")
KrizovaMaFy = load.img("faces/ucitele/KrizovaMaFy.jpg")
KvapilMaFyHv = load.img("faces/ucitele/KvapilMaFyHv.jpg")
LekesovaCjHv = load.img("faces/ucitele/LekesovaCjHv.jpg")
LetakovaBiCh = load.img("faces/ucitele/LetakovaBiCh.jpg")
MadrovaAj = load.img("faces/ucitele/MadrovaAj.jpg")
MarkovaBiCh = load.img("faces/ucitele/MarkovaBiCh.jpg")
MartinikovaNeHv = load.img("faces/ucitele/MartinikovaNeHv.jpg")
MasnyOvVv = load.img("faces/ucitele/MasnyOvVv.jpg")
McMullinAj = load.img("faces/ucitele/McMullinAj.jpg")
MenouskovaCjFr = load.img("faces/ucitele/MenouskovaCjFr.jpg")
MinarikMa = load.img("faces/ucitele/MinarikMa.jpg")
NavratilBiZ = load.img("faces/ucitele/NavratilBiZ.jpg")
NavratilovaAj = load.img("faces/ucitele/NavratilovaAj.jpg")
# nepritomni = load.img("faces/ucitele/nepritomni.txt")
NezvalovaAjRuD = load.img("faces/ucitele/NezvalovaAjRuD.jpg")
OndrakOvLat = load.img("faces/ucitele/OndrakOvLat.jpg")
PazderovaMaFy = load.img("faces/ucitele/PazderovaMaFy.jpg")
PetrMaFy = load.img("faces/ucitele/PetrMaFy.jpg")
PiechovaAj = load.img("faces/ucitele/PiechovaAj.jpg")
PohanelMaZAjProjekty = load.img("faces/ucitele/PohanelMaZAjProjekty.jpg")
PolednovaMaZ = load.img("faces/ucitele/PolednovaMaZ.jpg")
PridalovaAjRu = load.img("faces/ucitele/PridalovaAjRu.jpg")
PrzybylovaCjHv = load.img("faces/ucitele/PrzybylovaCjHv.jpg")
PuskarovaNeCj = load.img("faces/ucitele/PuskarovaNeCj.jpg")
RehulkaBiCh = load.img("faces/ucitele/RehulkaBiCh.jpg")
RichterkovaMaFy = load.img("faces/ucitele/RichterkovaMaFy.jpg")
SedlackovaCjFr = load.img("faces/ucitele/SedlackovaCjFr.jpg")
SeverovaMaIVT = load.img("faces/ucitele/SeverovaMaIVT.jpg")
SladecekCjAj = load.img("faces/ucitele/SladecekCjAj.jpg")
SoubustovaNeRu = load.img("faces/ucitele/SoubustovaNeRu.jpg")
StanecSp = load.img("faces/ucitele/StanecSp.jpg")
StepanovIVT = load.img("faces/ucitele/StepanovIVT.jpg")
StranskaFy = load.img("faces/ucitele/StranskaFy.jpg")
StudnickaMaIVT = load.img("faces/ucitele/StudnickaMaIVT.jpg")
SuchankovaAj = load.img("faces/ucitele/SuchankovaAj.jpg")
SykorovaAjD = load.img("faces/ucitele/SykorovaAjD.jpg")
TeplyAjTv = load.img("faces/ucitele/TeplyAjTv.jpg")
UrbaskovaMaZ = load.img("faces/ucitele/UrbaskovaMaZ.jpg")
VackovaCjVVEkonomie = load.img("faces/ucitele/VackovaCjVVEkonomie.jpg")
VasiljevovaBiVV = load.img("faces/ucitele/VasiljevovaBiVV.jpg")
VavraCjVV = load.img("faces/ucitele/VavraCjVV.jpg")
VeselaCjNe = load.img("faces/ucitele/VeselaCjNe.jpg")
VitamvasovaAj = load.img("faces/ucitele/VitamvasovaAj.jpg")
VojtovicovaMaFyAj = load.img("faces/ucitele/VojtovicovaMaFyAj.jpg")
ZabojovaBiCh = load.img("faces/ucitele/ZabojovaBiCh.jpg")
ZajicovaCjNe = load.img("faces/ucitele/ZajicovaCjNe.jpg")
ZarubovaMaCh = load.img("faces/ucitele/ZarubovaMaCh.jpg")
ZatloukalovaVVRu = load.img("faces/ucitele/ZatloukalovaVVRu.jpg")

known_face_encodings = [AbrahamovaTv, Adamikova2AjRuD, BednarikovaMaITV, BerankovaAjFr, BraunerChBi, CastellarSp, CechovaAj, ChromaTvZ, DanielTvZ, DohnalovaCjHv, DostalovaTv, DubovaNeCj, DvorskyAjD, FaberovaAjRu, FerencMaFy, FukaCjNe, GlosCjD, gosD, HajduMaAj, HajekBiCh, HamrikovaMaCh, HavranovaAj, HermanovaCjHvRu, HorvathSp, HubackovaCjD, JanickovaAjCj, JonesAj, JurikTvZ, KalandrovaCjOv, KnapekAjSp, KonecnaNeCj, KostkovaMaCh, KovarovaMaCh, KralovaAj, KrcovaMaZ, KrejcirovaSp, KrejcirTvZ, KrizovaMaFy, KvapilMaFyHv, LekesovaCjHv, LetakovaBiCh, MadrovaAj, MarkovaBiCh, MartinikovaNeHv, MasnyOvVv, McMullinAj, MenouskovaCjFr, MinarikMa, NavratilBiZ, NavratilovaAj, NezvalovaAjRuD, OndrakOvLat, PazderovaMaFy, PetrMaFy, PiechovaAj, PohanelMaZAjProjekty, PolednovaMaZ, PridalovaAjRu, PrzybylovaCjHv, PuskarovaNeCj, RehulkaBiCh, RichterkovaMaFy, SedlackovaCjFr, SeverovaMaIVT, SladecekCjAj, SoubustovaNeRu, StanecSp, StepanovIVT, StranskaFy, StudnickaMaIVT, SuchankovaAj, SykorovaAjD, TeplyAjTv, UrbaskovaMaZ, VackovaCjVVEkonomie, VasiljevovaBiVV, VavraCjVV, VeselaCjNe, VitamvasovaAj, VojtovicovaMaFyAj, ZabojovaBiCh, ZajicovaCjNe, ZarubovaMaCh, ZatloukalovaVVRu]
known_face_names = ['AbrahamovaTv', 'Adamikova2AjRuD', 'BednarikovaMaITV', 'BerankovaAjFr', 'BraunerChBi', 'CastellarSp', 'CechovaAj', 'ChromaTvZ', 'DanielTvZ', 'DohnalovaCjHv', 'DostalovaTv', 'DubovaNeCj', 'DvorskyAjD', 'FaberovaAjRu', 'FerencMaFy', 'FukaCjNe', 'GlosCjD', 'gosD', 'HajduMaAj', 'HajekBiCh', 'HamrikovaMaCh', 'HavranovaAj', 'HermanovaCjHvRu', 'HorvathSp', 'HubackovaCjD', 'JanickovaAjCj', 'JonesAj', 'JurikTvZ', 'KalandrovaCjOv', 'KnapekAjSp', 'KonecnaNeCj', 'KostkovaMaCh', 'KovarovaMaCh', 'KralovaAj', 'KrcovaMaZ', 'KrejcirovaSp', 'KrejcirTvZ', 'KrizovaMaFy', 'KvapilMaFyHv', 'LekesovaCjHv', 'LetakovaBiCh', 'MadrovaAj', 'MarkovaBiCh', 'MartinikovaNeHv', 'MasnyOvVv', 'McMullinAj', 'MenouskovaCjFr', 'MinarikMa', 'NavratilBiZ', 'NavratilovaAj', 'NezvalovaAjRuD', 'OndrakOvLat', 'PazderovaMaFy', 'PetrMaFy', 'PiechovaAj', 'PohanelMaZAjProjekty', 'PolednovaMaZ', 'PridalovaAjRu', 'PrzybylovaCjHv', 'PuskarovaNeCj', 'RehulkaBiCh', 'RichterkovaMaFy', 'SedlackovaCjFr', 'SeverovaMaIVT', 'SladecekCjAj', 'SoubustovaNeRu', 'StanecSp', 'StepanovIVT', 'StranskaFy', 'StudnickaMaIVT', 'SuchankovaAj', 'SykorovaAjD', 'TeplyAjTv', 'UrbaskovaMaZ', 'VackovaCjVVEkonomie', 'VasiljevovaBiVV', 'VavraCjVV', 'VeselaCjNe', 'VitamvasovaAj', 'VojtovicovaMaFyAj', 'ZabojovaBiCh', 'ZajicovaCjNe', 'ZarubovaMaCh', 'ZatloukalovaVVRu']




# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
print("variables")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # print(ret)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # small_frame = frame

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
