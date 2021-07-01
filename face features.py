from scipy.stats import pearsonr
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import numpy as np
import dlib
import cv2
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)
count=0
while True:
 frame = vs.read()
 frame = imutils.resize(frame, width=400)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 rects = detector(gray, 0)
 for rect in rects:
     shape = predictor(gray, rect)
     shape = face_utils.shape_to_np(shape)

     for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
       print name
       for (x, y) in shape[i:j]:
         cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
         (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
         roi = frame[y:y + h, x:x + w]
         roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
         height, width = roi.shape[:2]
         cv2.circle(roi, (int(width)/2, int(height)/2), 2, (0, 0, 255), -1)
         new=[roi.size, int(height), int(width), int(height * width), float(width) / float(height)]
 cv2.imshow("Frame", frame)
 key = cv2.waitKey(1) & 0xFF
 if key == ord("q"):
     break

vs.stop()
cv2.destroyAllWindows()
