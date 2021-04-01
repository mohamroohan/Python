import numpy as np
import imutils
import cv2
import time
from imutils.video import VideoStream
faceCascade = cv2.CascadeClassifier('D:\PROGRAMS\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
leftStream = VideoStream(src=0).start()
while True:
    frame = leftStream.read()
    if frame is not None:
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 2)
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
leftStream = VideoStream(src=0).stop()
cv2.destroyAllWindows()