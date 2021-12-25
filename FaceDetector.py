import cv2
import numpy as np
import dlib

# values

# dimension 1
# 0 - face
# 1 - left eye
# 2 - left pupil
# 3 - right eye
# 4 - right pupil

# dimension 2
# 0 - location x
# 1 - location y
# 2 - size

# for training threshold
MAX_TRAIN_ITERATIONS = 200 # try threshold from 0 to this value
MIN_PUPIL_AREA = 0.1 # the training is finished, if pupil-area / eye-area > this value

class FaceDetector:
    MAX_LOST = 5 # maximum amount of frames without detection of an object, before it is lost
    
    def __init__(self):
        self.image = None # given image to detect the face
        self.imageGray = None # grayScale-image of the image
        
        self.values = np.zeros((5,3)) # see definition at the top
        self.lost = np.ones((5,1)) * self.MAX_LOST # array of lost values
        
        self.detector = dlib.get_frontal_face_detector() # face detector from dlib
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # shape predictor from dlib
        self.landmarks = None # detected landmarks
        
        self.threshold = 50 # starting threshold for pupil detection
        
    # mark the given object (location, size) in the given image
    def _markObject(self, image, index, name, color, rectangle):
        if(self.lost[index,0] < self.MAX_LOST):
            if(rectangle):
                x1 = int(round(self.values[index,0] - self.values[index,2]))
                y1 = int(round(self.values[index,1] - self.values[index,2]))
                x2 = int(round(self.values[index,0] + self.values[index,2]))
                y2 = int(round(self.values[index,1] + self.values[index,2]))
                cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
            else:
                x = int(round(self.values[index,0]))
                y = int(round(self.values[index,1]))
                rad = int(round(self.values[index,2]))
                cv2.circle(image, (x,y), rad, color, 1)     
            if(name is not None):
                x = int(round(self.values[index,0] - self.values[index,2]))
                y = int(round(self.values[index,1] - self.values[index,2] - 5))
                fontScale = 0.4
                thickness = 1
                cv2.putText(image, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)    
        
    # detect the face
    def _detectFace(self):
        faces = self.detector(self.imageGray) # detect faces in the image
        if(len(faces) > 0): # check for detected faces
            face = faces[0] # use the first found face
            self.landmarks = self.predictor(self.imageGray, face) # detect the landmarks of the face
            self.lost[0,0] = 0
            self.values[0,2] = (face.right()-face.left())/2
            self.values[0,0] = face.left() + self.values[0,2]
            self.values[0,1] = face.top() + self.values[0,2]
        else:
            self.lost[0,0] += 1
        
    # detect the given eye
    def _detectEye(self, left):
        index = 3 # index of right eye in the values list
        partLeft = 42 # left landmark of the right eye
        partRight = 45 # right landmark of the right eye
        if(left):
            index = 1
            partLeft = 36
            partRight = 39
        if(self.lost[0,0] < self.MAX_LOST):
            self.lost[index,0] = 0
            
            x1 = self.landmarks.part(partLeft).x
            y1 = self.landmarks.part(partLeft).y
            x2 = self.landmarks.part(partRight).x
            y2 = self.landmarks.part(partRight).y
    
            self.values[index,2] = (x2-x1)/2*1.2
            self.values[index,0] = (x1+x2)/2
            self.values[index,1] = (y1+y2)/2
        else:
            self.lost[index,0] = self.MAX_LOST
                 
    # detect the given pupil
    def _detectPupil(self, left):
        indexEye = 3 # index of right eye in the values list
        indexPupil = 4 # index of right pupil in the values list
        if(left):
            indexEye = 1
            indexPupil = 2
        if(self.lost[indexEye,0] < self.MAX_LOST):
            x1 = self.values[indexEye,0] - self.values[indexEye,2]
            y1 = self.values[indexEye,1] - self.values[indexEye,2]
            x2 = self.values[indexEye,0] + self.values[indexEye,2]
            y2 = self.values[indexEye,1] + self.values[indexEye,2]
            # get sector of the image of the given eye (red color channel, best results for pupils)
            imageBinary = self.image[int(round(y1)):int(round(y2)),int(round(x1)):int(round(x2)),2]
            _, imageBinary = cv2.threshold(imageBinary, self.threshold, 255, cv2.THRESH_BINARY)
            
            keypoints = np.where(imageBinary == 0)
            if(len(keypoints[0]) > 0):
                # calculate center of used pixels as pupil location
                xmean = np.mean(keypoints[1])
                ymean = np.mean(keypoints[0])
                self.lost[indexPupil,0] = 0
                self.values[indexPupil,0] = x1 + xmean # correct relative position, as image is only a sector
                self.values[indexPupil,1] = y1 + ymean
                self.values[indexPupil,2] = 5
            else:
                self.lost[indexPupil,0] += 1
        else:
            self.lost[indexPupil,0] = self.MAX_LOST       
            
    # detect a face in the given image
    def detect(self, image):
        self.image = image
        self.imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self._detectFace()
        self._detectEye(True)
        self._detectPupil(True)
        self._detectEye(False)
        self._detectPupil(False)
        
    # train the threshold by increasing it starting at 0, until a minimum pupil area is reached
    def trainThreshold(self):
        self.threshold = 0
        while(self.threshold < MAX_TRAIN_ITERATIONS and self.threshold < 255):
            image = self.getBinaryEyeImage(True)
            if(image is not None):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                values = image.shape[0] * image.shape[1]
                area = values - np.sum(image) / 255
                if(area / values >= MIN_PUPIL_AREA):
                    break
            self.threshold += 1
    
    # returns an image, where all detected objects are marked
    def getMarkedImage(self):
        image = np.copy(self.image)
        self._markObject(image, 0, 'Face', (0,0,0), True)
        self._markObject(image, 1, 'Left Eye', (0,0,0), True)
        self._markObject(image, 2, None, (0,0,255), False)
        self._markObject(image, 3, 'Right Eye', (0,0,0), True)
        self._markObject(image, 4, None, (0,0,255), False)
        return image
    
    # returns the binary image used to determine the center of the pupil
    def getBinaryEyeImage(self, left):
        index = 3
        if(left):
            index = 1
        if(self.lost[index,0] < self.MAX_LOST):
            x1 = self.values[index,0] - self.values[index,2]
            y1 = self.values[index,1] - self.values[index,2]
            x2 = self.values[index,0] + self.values[index,2]
            y2 = self.values[index,1] + self.values[index,2]
            imageBinary = self.image[int(round(y1)):int(round(y2)),int(round(x1)):int(round(x2)),2]
            _, imageBinary = cv2.threshold(imageBinary, self.threshold, 255, cv2.THRESH_BINARY)
            imageBinary = cv2.cvtColor(imageBinary, cv2.COLOR_GRAY2BGR)
            return imageBinary
        return None     