import numpy as np
import cv2
from tkinter import MessageBox
from tkinter import *
root=Tk()
root.withdraw()
def detection():
   webcam = cv2.VideoCapture(0)
   if webcam.isOpened():
    classifier1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while(1):
        (rval, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = classifier1.detectMultiScale(gray,1.3,5)
        for (x,y,h,w) in faces:
          cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
          print(x,y,h,w)
        cv2.imshow('User Registeration', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
            cv2.destroyAllWindows()
   else:
        showerror("Error", "Please Connect your webam")
detection()
