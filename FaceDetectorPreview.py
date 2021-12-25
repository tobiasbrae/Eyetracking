import cv2

import FaceDetector

videoCapture = cv2.VideoCapture(0)
detector = FaceDetector.FaceDetector()

returnValue, image = videoCapture.read()

while(returnValue == 1):            
    detector.detect(image)
    
    outputImage = detector.getMarkedImage()
    cv2.imshow('FaceDetectorPreview', outputImage)
    
    outputImage2 = detector.getBinaryEyeImage(True)
    if(outputImage2 is not None):
        cv2.imshow('Eye', outputImage2)
    
    # read and handle pressed keys
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break
    
    returnValue, image = videoCapture.read()
        
cv2.destroyAllWindows()