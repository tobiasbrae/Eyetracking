import cv2
import numpy as np
import dlib

image = cv2.imread('face.png')
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
faces = detector(imageGray)

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
landmarks = predictor(imageGray, faces[0])

sizeX = image.shape[1]
sizeY = image.shape[0]

outputImage = np.ones((sizeY, sizeX, 3)) * 255

for i in range(68):
    x = landmarks.part(i).x
    y = landmarks.part(i).y
    radius = 1
    color = (0, 0, 0)
    thickness = 5
    cv2.circle(outputImage, (x,y), radius, color, thickness)
    
    x2 = x + 5
    y2 = y - 5
    fontScale = 0.4
    thickness = 1
    cv2.putText(outputImage, str(i), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)


cv2.imshow('Output', outputImage)

while(cv2.waitKey(1) != ord('q')):
    a = 1
    
cv2.destroyAllWindows()