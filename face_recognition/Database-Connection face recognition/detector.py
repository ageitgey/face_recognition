from sqlite3.dbapi2 import Cursor
import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3


recognizer= cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath="Classifiers/face.xml"
faceCascade=cv2.CascadeClassifier(cascadePath)
path='dataset'



def getProfile(id):
    conn=sqlite3.connect("faceBase.py")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


cam=cv2.VideoCapture(1)
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1,0, 1, 1)

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataset/user."+id+'.'+str(sampleNum)+"jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(im , (x-50,y-50), (x+w+50,y+h+50),(225,0,0),2)

    cv2.imshow('im',im)
    cv2.waitKey(100)
    if sampleNum>20:
        cam.realease()
        cv2.destroyAllWindows()
        break






