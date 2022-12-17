# Import
import pathlib
import cv2

# Pointing to dataset path
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt2.xml"

# Building classifier
cascade_classifier = cv2.CascadeClassifier(str(cascade_path))

# Setting default camera
default_camera = cv2.VideoCapture(0)

# Taking input from camera to detect image
while True:
    _, frame = default_camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces_input = cascade_classifier.detectMultiScale(
        # scaleFactor set from 1.1, minNeighbors set from 10, minSize set from (30, 30), change color to RGB2GRAY
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Plot to showcase face being detected
    for (x, y, width, height) in faces_input:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 512)

    cv2.imshow("Facy", frame)
    if cv2.waitKey(1) == ord("q"):
        break

default_camera.release()
cv2.destroyAllWindows()
