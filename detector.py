import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3


recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path='dataSet'
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


cam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,225,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.cv.PutText(cv2.cv.fromarray(im),str(profile[1]), (x,y+h+30),font, (0,225,0));
            cv2.cv.PutText(cv2.cv.fromarray(im),str(profile[2]), (x,y+h+60),font, (0,225,0));
            cv2.cv.PutText(cv2.cv.fromarray(im),str(profile[3]), (x,y+h+90),font, (0,225,0));


            
        
    cv2.imshow('im',im); 
    if cv2.waitKey(10) ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
