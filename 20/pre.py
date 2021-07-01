
#used to find euclidean distance betwwen two points
from scipy.spatial import distance as dist
from imutils.video import VideoStream
#identify the face
from imutils import face_utils
from threading import Thread
import numpy as np
#play sound
import playsound
#detect edges
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play an alarm sound
	print ("blinked")
	playsound.playsound(path)



#centre point of eye 
EYE_AR_THRESH = 0.2
 
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def Detection():
        # start the video stream thread
        print("[INFO] starting video stream thread...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

        # loop over frames from the video stream
        while True:
                
                frame = vs.read()
                frame = imutils.resize(frame, width=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale frame
                rects = detector(gray, 0)

                # loop over the face detections
                for rect in rects:
                        
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        

                        #extract left eye and right eye,draw points
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                        break


        cv2.destroyAllWindows()
        vs.stop()

detection()        
