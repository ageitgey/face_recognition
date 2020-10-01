import os
import cv2
import face_recognition
import numpy as np
import pyttsx3

engine = pyttsx3.init()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    engine.stop

path = r'path'#give the path to the folder where your images are
images = []
names = []
myfiles = os.listdir(path)

for img in myfiles:
    curimg = cv2.imread(f'{path}\{img}')
    images.append(curimg)
    names.append(os.path.splitext(img)[0])

authorised = ['rohan','mini' , 'rajesh' , 'neelam' , 'rani']
namesrohan = ['rohu' , 'rohan sir' , 'daksh' , 'hero']



def findencodeings(images):
    encodelist = []
    for i in images:
        i = cv2.cvtColor(i , cv2.COLOR_BGR2RGB)
        currencode = face_recognition.face_encodings(i)[0]
        encodelist.append(currencode)
    return encodelist

encodelistknown = findencodeings(images)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faceframe = face_recognition.face_locations(frame)
    faceencodes = face_recognition.face_encodings(frame , faceframe)
    for faceloc in faceframe:
        y1,x2,y2,x1 = faceloc
    for encodes in faceencodes:
        match = face_recognition.compare_faces(encodelistknown , encodes , tolerance=0.4)
        facedis = face_recognition.face_distance(encodelistknown , encodes)
        matchind = np.argmin(facedis)
        if match[matchind]:
            name = names[matchind]
            if name.lower() in namesrohan:
                name  = name.replace(name , 'Rohan')

            if name.lower() in authorised:
                frame = cv2.putText(frame , name , (x1+10 , y1-35) , cv2.FONT_ITALIC , 2 , (255,255,0) , 2)
                frame = cv2.putText(frame , '(authorised)' , (x1+10 , y1-10) , cv2.FONT_ITALIC , 1 , (0,255,0) , 2)
                frame = cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , (0,255,0) , 2)
                cv2.imshow('video' , frame)
                speak(f'Face detected you are {name}')
                quit()
        else:
            frame = cv2.putText(frame , '(not authorised)' , (x1+10 , y1-10) , cv2.FONT_ITALIC , 1 , (0,0,255) , 2)
            frame = cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , (0,0,255) , 2)
            os.system('shutdown -s -t 3')

    cv2.imshow('video' , frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
