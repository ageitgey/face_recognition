import cv2
import numpy as np
import sqlite3

# Load the cascade
from Tools.scripts.treesync import raw_input

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')


def insertOrupdate(Id, Name):
    conn=sqlite3.connect("faceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1

    if(isRecordExist==1):
        cmd="UPDATE People SET NAME "+str(Name)+"WHERE ID"+str(Id)
    else:
        cmd="INSERT INTO People(ID,NAME) Values("+str(Id)+","+str(Name)+")"

    conn.execute(cmd)
    conn.commit()
    conn.close()

id = raw_input('enter user id')
name = raw_input('enter your name')
insertOrupdate(id,name)
sampleNum = 0

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataSet/User." +str(id)+"." +str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(100)

    cv2.imshow('img', img)
    cv2.waitKey(1)

    if (sampleNum>20):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()