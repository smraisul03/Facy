# June 7, 2022, Lab
# Face Detection Live

# Imports
import numpy as np
import cv2
import time

# Create face classifier object
face_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create eye classifier object
eye_classifier = cv2.CascadeClassifier('data/haarcascade_eye.xml')


# Function will detect both face and eyes
def face_eye_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Classifier return the ROI of detected face as a tuple
    # If face is found, returns array for positions of detected face as Rect(x,y,w,h)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # When no face is detected, return empty tuple
    if faces is ():
        return img

    # Iterate through face array and draw rectangle over each faces in face
    for (x, y, w, h) in faces:
        # Check to see if rectangle is bigger than face
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50

        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Crop face out of image
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]

        # Once location is found, create ROI for face
        # Apply eye detection on ROI
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey, + eh), (0, 0, 255), 2)

        # roi_color = cv2.flip(roi_color, 1)

        # Show cropped image
        return roi_color


# Init camera
cap = cv2.VideoCapture(0)

# Loop frame
while True:
    ret, frame = cap.read()
    cv2.imshow("Live Face And Eye Extractor", face_eye_detector(frame))

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()