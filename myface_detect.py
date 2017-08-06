
import cv2      #openCV module in python
import sys

###  Taking input from the user as command line args
imgPath = sys.argv[1]
cascPath = 'haarcascade_frontalface_default.xml'

###  Cascade classifier performs Machine Learning series of tests to detect
###  if a frame is having the features of face or not.
faceCascade = cv2.CascadeClassifier(cascPath)

###  reads the image at the given path
image = cv2.imread(imgPath)

###  converts image from BGR format to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

###  detectMultiScale is a general function to detect the objects in the given frame.
###  Using faceDetect, this function is detecting the faces 
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minSize=(30, 30),
    minNeighbors=5,
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE         
)


print 'Found %s faces.' % len(faces)        #number of faces found in the picture

###  Draws a rectangle around the faces (that are found) of given dimensions
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

###  shows the editted image on the screen
cv2.imshow('Faces Found', image)

###  waits for the user to press any key to close the window 
cv2.waitKey(0)
