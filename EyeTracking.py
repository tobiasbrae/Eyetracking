import cv2
import tkinter
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import pickle
import os

import FaceDetector

# device id for used webcam
DEVICE_ID = 0

# types for render-objects
TYPE_CIRCLE = 'circle'
TYPE_TEXT = 'text'
TYPE_IMAGE = 'image'
TYPE_LINE = 'line'

# colors
COLOR_BACKGROUND = (0,0,0)
COLOR_TEXT = (255,255,255)
COLOR_CURSOR = (0,128,255)
COLOR_MEAN = (0,255,0)
COLOR_TIME =[(255,0,255),(0,255,255)]
COLOR_GRID = (100,100,100)

PRINT_FRIENDLY = False

if(PRINT_FRIENDLY):
    COLOR_BACKGROUND = (255,255,255)
    COLOR_TEXT = (0,0,0)
    COLOR_CURSOR = (0,100,200)
    COLOR_MEAN = (0,255,0)
    COLOR_TIME =[(255,0,255),(0,255,255)]
    COLOR_GRID = (100,100,100)

# operating mode
MODE_NORMAL = 0
MODE_CAMERA_SIZE = 1
MODE_THRESHOLD = 2
MODE_CALIBRATE_CURSOR = 3

# mean filter
MEAN_AMOUNT = 5 # number of used locations

# time based filter
TIME_STEP = 0.1

# grid filter
GRID_ROWS = 5
GRID_COLS = 7
GRID_MAX = 25
GRID_MIN = 2

#defined keys
KEY_HELP = 'h'
KEY_QUIT = 'q'
KEY_CALIBRATE = 'c'
KEY_RESET = 'r'
KEY_MEAN = 'm'
KEY_TIME = 't'
KEY_GRID = 'g'
KEY_NEXT = 'n'
KEY_INC = '+'
KEY_DEC = '-'
KEY_SAVE = 's'
KEY_LOAD = 'l'
KEY_ROWS_INC = '8'
KEY_ROWS_DEC = '2'
KEY_COLS_INC = '6'
KEY_COLS_DEC = '4'
KEY_USE_HEAD = 'u'

class RenderObject:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.x2 = 0
        self.y2 = 0
        self.width = 0
        self.height = 0
        self.type = type
        self.radius = 10
        self.color = COLOR_TEXT
        self.fontScale = 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text = 'Text'
        self.thickness = 1
        self.visible = True
        self.deleteAfter = 0
        self.image = None
        self.arrange = False
        
    # determine the size of a text-type renderObject
    def determineSize(self):
        if(self.type == TYPE_TEXT):
            text_size = cv2.getTextSize(self.text, self.font, self.fontScale, self.thickness)
            self.width = text_size[0][0]
            self.height = text_size[0][1]
            
    # render the RenderObject on the given image
    def render(self, image):
        if(self.visible):
            x = int(round(self.x))
            y = int(round(self.y))
            if(self.type == TYPE_CIRCLE):
                radius = int(round(self.radius))
                cv2.circle(image, (x, y), radius, self.color, self.thickness)
            elif(self.type == TYPE_TEXT):
                cv2.putText(image, self.text, (x, y), self.font, self.fontScale, self.color, self.thickness)
            elif(self.type == TYPE_IMAGE):
                if(self.image is not None):
                    if(self.image.shape[0] != self.height or self.image.shape[1] != self.width):
                        if(self.height == 0 or self.width == 0):
                            self.height = self.image.shape[0]
                            self.width = self.image.shape[1]
                        else:
                            self.image = cv2.resize(self.image, (self.width, self.height))
                    width = image.shape[1]
                    height = image.shape[0]
                    if(x >= 0 and x < width and y >= 0 and y < height):
                        x1 = x
                        y1 = y
                        x2 = min(width-1, x1 + self.width)
                        y2 = min(height-1, y1 + self.height)
                        image[y1:y2,x1:x2] = self.image[0:y2-y1,0:x2-x1]
            elif(self.type == TYPE_LINE):
                x2 = int(round(self.x2))
                y2 = int(round(self.y2))
                cv2.line(image, (x,y), (x2,y2), self.color, self.thickness)

class EyeTracking:
    def __init__(self):
        # operation mode
        self.mode = MODE_NORMAL
        
        # window settings
        self.windowName = 'EyeTracking'
        self.root = tkinter.Tk()
        self.width = self.root.winfo_screenwidth() # width of the screen in pixels
        self.height = self.root.winfo_screenheight() # height of the screen in pixels
        self.x = int(round(self.width/2)) # center of the screen (x-axis)
        self.y = int(round(self.height/2)) # center of the screen (y-axis)
        
        # face detector
        self.detector = FaceDetector.FaceDetector()
        self.values = None
        self.lost = None
        
        # calibration values
        self.locations = []
        self.initLocations()
        self.calibrationData = []
        self.calibrationDataBackup = []
        self.models = []
        self.models.append(LinearRegression())
        self.models.append(LinearRegression())
        self.models.append(LinearRegression())
        self.models.append(LinearRegression())
        self.calibrated = False
        
        # render options
        self.renderList = []
        self.arrangeOffset = 20
        
        # use head movement
        self.useHead = True
        
         # mean filter
        self.useFilterMean = False
        self.renderListMean = []
        self.meanList = []
        
        #time filter
        self.useFilterTime = False
        self.renderListTime = []
        
        # grid filter
        self.useFilterGrid = False
        self.renderListGrid = []
        self.gridRows = GRID_ROWS
        self.gridCols = GRID_COLS
        
        # filters
        self.createRenderListFilter()
        
        # image read from the webcam
        self.inputImage = None
        
        # cursor for calibration/visualization
        self.cursor = RenderObject(0, 0, TYPE_CIRCLE)
        self.cursor.radius = 15
        self.cursor.color = COLOR_CURSOR
        self.renderList.append(self.cursor)
        self.cursorLocation = 0
        
        # renderobject for showing images
        self.outputImage = RenderObject(0, 0, TYPE_IMAGE)
        self.outputImage.visible = False
        self.pushObject(self.outputImage)
        
        # scaling for webcam preview
        self.outputScale = 1.0
        
        # static information labels
        self.labelFace = RenderObject(5,50,TYPE_TEXT)
        self.labelFace.fontScale = 0.5
        self.renderList.append(self.labelFace)
        
        self.labelLeftEye = RenderObject(5,75,TYPE_TEXT)
        self.labelLeftEye.fontScale = 0.5
        self.renderList.append(self.labelLeftEye)
        
        self.labelRightEye = RenderObject(5,100,TYPE_TEXT)
        self.labelRightEye.fontScale = 0.5
        self.renderList.append(self.labelRightEye)
        
        self.labelThreshold = RenderObject(5,125,TYPE_TEXT)
        self.labelThreshold.fontScale = 0.5
        self.renderList.append(self.labelThreshold)
        
        self.labelFilterMean = RenderObject(5,150,TYPE_TEXT)
        self.labelFilterMean.fontScale = 0.5
        self.renderList.append(self.labelFilterMean)
        
        self.labelFilterTime = RenderObject(5,175,TYPE_TEXT)
        self.labelFilterTime.fontScale = 0.5
        self.renderList.append(self.labelFilterTime)
        
        self.labelFilterGrid = RenderObject(5,200,TYPE_TEXT)
        self.labelFilterGrid.fontScale = 0.5
        self.renderList.append(self.labelFilterGrid)
        
        self.labelFPS = RenderObject(5,225,TYPE_TEXT)
        self.labelFPS.fontScale = 0.5
        self.renderList.append(self.labelFPS)
        self.FPS = 0
        
        self.labelHead = RenderObject(5,250,TYPE_TEXT)
        self.labelHead.fontScale = 0.5
        self.renderList.append(self.labelHead)
        
        # welcome messages
        self.pushText('Welcome to the EyeTracker!', 100)
        self.pushText('Push <' + KEY_HELP + '> for help.', 150)
        
    # create list of screen locations for calibrating the cursor
    def initLocations(self):
        relativeLocations = []
        relativeLocations.append((0.5, 0.5))
        relativeLocations.append((0.1, 0.5))
        relativeLocations.append((0.1, 0.1))
        relativeLocations.append((0.5, 0.1))
        relativeLocations.append((0.9, 0.1))
        relativeLocations.append((0.9, 0.5))
        relativeLocations.append((0.9, 0.9))
        relativeLocations.append((0.5, 0.9))
        relativeLocations.append((0.1, 0.9))
        relativeLocations = np.array(relativeLocations)
        
        size = np.array((self.width,self.height))
        
        self.locations = relativeLocations * size
        
    # create renderObjects to visualize filters
    def createRenderListFilter(self):
        # create points for mean filter
        for i in range(MEAN_AMOUNT):
            point = RenderObject(0, 0, TYPE_CIRCLE)
            point.radius = 2
            point.color = COLOR_MEAN
            point.thickness = 2
            point.visible = False
            self.renderListMean.append(point)
            self.renderList.append(point)
            
        # create points for time filter
        for i in range(2):
            point = RenderObject(0, 0, TYPE_CIRCLE)
            point.radius = 2
            point.color = COLOR_TIME[i]
            point.thickness = 2
            point.visible = False
            self.renderListTime.append(point)
            self.renderList.append(point)
            
        # create grid array
        self.createRenderListGrid()
            
    # create grid array
    def createRenderListGrid(self):
        for obj in self.renderListGrid:
            self.renderList.remove(obj)
        self.renderListGrid = []
        
        # create lines for grid filter
        stepX = int(round(self.width / self.gridCols))
        for i in range(1,self.gridCols):
            line = RenderObject(stepX*i, 0, TYPE_LINE)
            line.x2 = stepX*i
            line.y2 = self.height
            line.color = COLOR_GRID
            line.visible = False
            line.thickness = 1
            self.renderListGrid.append(line)
            
        stepY = int(round(self.height / self.gridRows))
        for i in range(1,self.gridRows):
            line = RenderObject(0, stepY*i, TYPE_LINE)
            line.x2 = self.width
            line.y2 = stepY*i
            line.color = COLOR_GRID
            line.visible = False
            self.renderListGrid.append(line)
        
        for obj in self.renderListGrid:
            self.renderList.append(obj)
        
        
    # add renderObject to renderList, which automatically get's centered
    def pushObject(self, obj, deleteAfter=0):
        obj.arrange = True
        obj.deleteAfter = deleteAfter
        self.renderList.append(obj)
        
    # same a pushObject(...) but optimized for text
    def pushText(self, text, deleteAfter=0):
        for obj in self.renderList:
            if(obj.text == text):
                return False
        obj = RenderObject(0, 0, TYPE_TEXT)
        obj.text = text
        self.pushObject(obj, deleteAfter)
    
    # show a list of available commands
    def showCommandList(self):
        self.pushText('Push <' + KEY_QUIT + '> to quit the EyeTracker.', 150)
        self.pushText('Push <' + KEY_CALIBRATE + '> to start a calibration.', 150)
        self.pushText('Push <' + KEY_RESET + '> to reset the calibration values.', 150)
        self.pushText('Push <' + KEY_MEAN + '> to toggle mean filter.', 150)
        self.pushText('Push <' + KEY_TIME + '> to toggle time filter.', 150)
        self.pushText('Push <' + KEY_GRID + '> to toggle grid filter.', 150)
        self.pushText('Push <' + KEY_SAVE + '> to save calibration data.', 150)
        self.pushText('Push <' + KEY_LOAD + '> to load calibration data.', 150)
        self.pushText('Push <' + KEY_USE_HEAD + '> to toggle usage of head position.', 150)
        
    # calculate locations for renderObjects that get arranged in the center
    def arrangePushedObjects(self):
        height = 0
        for obj in self.renderList:
            if(obj.visible and obj.arrange):
                obj.determineSize()
                height += obj.height + self.arrangeOffset
        y = int(round(self.y - height / 2))
        for obj in self.renderList:
            if(obj.visible and obj.arrange):
                obj.x = int(round(self.x - obj.width / 2))
                obj.y = y
                y += obj.height + self.arrangeOffset
                if(obj.type == TYPE_IMAGE):
                    y += 2 * self.arrangeOffset
                
    # update the texts of the information labels
    def updateLabels(self):
        if(self.values is not None):
            self.labelFace.text = 'Face lost: {} frames'.format(int(self.lost[0,0]))
            self.labelLeftEye.text = 'Left Eye lost: {} frames'.format(int(self.lost[2,0]))
            self.labelRightEye.text = 'Right Eye lost: {} frames'.format(int(self.lost[4,0]))
        
        self.labelThreshold.text = 'Threshold: {}'.format(self.detector.threshold)
        self.labelFilterMean.text = 'Filter Mean: {}'.format(self.useFilterMean)
        self.labelFilterTime.text = 'Filter Time: {}'.format(self.useFilterTime)
        self.labelFilterGrid.text = 'Filter Grid: {}'.format(self.useFilterGrid)
        self.labelFPS.text = 'FPS: {}'.format(self.FPS)
        self.labelHead.text = 'Use Head location: {}'.format(self.useHead)
    
    # render the current output image
    def renderImage(self):
        #create empty image
        backgroundColor = np.array(COLOR_BACKGROUND, dtype=np.uint8)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:,:] = backgroundColor
        
        # prepare renderObjects
        self.arrangePushedObjects()
        self.updateLabels()
        
        # render each object
        for obj in self.renderList:
            obj.render(image)
            # handle automatic deletion of the objects
            if(obj.deleteAfter > 0):
                obj.deleteAfter -= 1
                if(obj.deleteAfter == 0):
                    self.renderList.remove(obj)
        
        #return rendered image
        return image
    
    # immediately remove all renderObjects that are meant to be removed
    def removeRenderObjects(self):
        deletionList = [] # create list ob objects to be deleted
        for obj in self.renderList:
            if(obj.deleteAfter > 0): # gather all objects that are to be deleted
                deletionList.append(obj)
        for obj in deletionList: # finally delete objects
            self.renderList.remove(obj)
        # deletion list is needed as remove disturbs the for-loop
                
    # calculate cursor location based on operating mode
    def locateCursor(self):
        if(self.mode == MODE_NORMAL):
            self.locateNormalCursor()
        elif(self.mode == MODE_CALIBRATE_CURSOR):
            self.locateCalibrationCursor()
            
    # locate calibrated cursor in normal mode
    def locateNormalCursor(self):
        if(self.calibrated):
            # create array of current input values
            inputValues = []
            inputValues.append((self.values[2,0]-self.values[1,0],self.values[1,0]))
            inputValues.append((self.values[2,1]-self.values[1,1],self.values[1,1]))
            inputValues.append((self.values[4,0]-self.values[3,0],self.values[3,0]))
            inputValues.append((self.values[4,1]-self.values[3,1],self.values[3,1]))
            inputValues = np.array(inputValues)
            
            # create array of output values
            outputValues = np.zeros((4,1))
            for i in range(4):
                # add offset
                outputValues[i] = self.models[i].intercept_
                # calculate each multiplication
                for j in range(len(self.models[i].coef_)):
                    outputValues[i] += inputValues[i,j]*self.models[i].coef_[j]
            
            # mean values of both eyes
            newX = (outputValues[0,0] + outputValues[2,0])/2
            newY = (outputValues[1,0] + outputValues[3,0])/2
            
            # limit to frame borders
            newX = max(newX, self.cursor.radius)
            newX = min(newX, self.width-self.cursor.radius)
            newY = max(newY, self.cursor.radius)
            newY = min(newY, self.height-self.cursor.radius)
            
            # filter functions
            [newX, newY] = self.filterMean(newX, newY)
            [newX, newY] = self.filterTime(newX, newY)
            [newX, newY] = self.filterGrid(newX, newY)
            
            # write location to cursor
            self.cursor.x = newX
            self.cursor.y = newY
            
    # filter cursor location by taking mean value over the past n locations
    def filterMean(self, newX, newY):
        self.meanList.append((newX, newY))
        if(len(self.meanList) > MEAN_AMOUNT):
            self.meanList.pop(0)    
        # copy locations of used locations to the visualized points
        for i in range(min(len(self.meanList), MEAN_AMOUNT)):
            self.renderListMean[i].x = self.meanList[i][0]
            self.renderListMean[i].y = self.meanList[i][1]
        # compute filter
        if(self.useFilterMean):
            meanList = np.array(self.meanList)
            meanList = np.mean(meanList, 0)
            newX = meanList[0]
            newY = meanList[1]
        return [newX, newY]
        
    # filter cursor location by only moving cursor towards target
    def filterTime(self, newX, newY):
        if(self.useFilterTime):
            # copy current and target location to the visualized points
            self.renderListTime[0].x = self.cursor.x
            self.renderListTime[0].y = self.cursor.y
            self.renderListTime[1].x = newX
            self.renderListTime[1].y = newY
            # calculate new cursor position
            diffX = newX - self.cursor.x
            diffY = newY - self.cursor.y
            newX = self.cursor.x + diffX * TIME_STEP
            newY = self.cursor.y + diffY * TIME_STEP
        return [newX, newY]
    
    # filter cursor location by creating a grid    
    def filterGrid(self, newX, newY):
        if(self.useFilterGrid):
            width = self.width / self.gridCols
            height = self.height / self.gridRows
            col = int(newX/width)
            row = int(newY/height)
            newX = (col+0.5)*width
            newY = (row+0.5)*height
        return [newX, newY]
        
    # locate cursor while calibrating
    def locateCalibrationCursor(self):
        self.cursor.visible = True
        self.cursor.x = self.locations[self.cursorLocation,0]
        self.cursor.y = self.locations[self.cursorLocation,1]
        
    # fit the models after a calibration is finished
    def computeCalibration(self):
        if(len(self.calibrationData) > 0):
            self.computeRegression(0,2,0) # left eye, x axis
            self.computeRegression(1,2,1) # left eye, y axis
            self.computeRegression(2,4,0) # right eye, x axis
            self.computeRegression(3,4,1) # right eye, y axis
        
    # fit the given model by using the given index and axis
    def computeRegression(self, model, index, axis):
        # create vectors
        x = []
        y = []
        
        # add each location to vectors
        for i in range(len(self.calibrationData)):
            if(self.useHead):
                x.append((self.calibrationData[i][2][index,axis]-self.calibrationData[i][2][index-1,axis], self.calibrationData[i][2][index-1,axis]))
            else:
                x.append((self.calibrationData[i][2][index,axis]-self.calibrationData[i][2][index-1,axis], 0))
            y.append(self.calibrationData[i][axis])
            
        x = np.array(x)
        y = np.array(y)
        
        # fit the model
        self.models[model].fit(x,y)
        
        print('{:.2f}, {:.2f}, {:.2f}'.format(self.models[model].coef_[1], self.models[model].coef_[0], self.models[model].intercept_))
        
    # determine visibilities
    def determineVisibility(self):
        # determine visibility of cursor
        if((self.mode == MODE_NORMAL and self.calibrated) or self.mode == MODE_CALIBRATE_CURSOR):
            self.cursor.visible = True
        else:
            self.cursor.visible = False
            
        # determine visibility of outputImage
        if(self.mode == MODE_CAMERA_SIZE or self. mode == MODE_THRESHOLD):
            self.outputImage.visible = True
        else:
            self.outputImage.visible = False
            
        # determine visibility of mean points
        visible = False
        if(self.mode == MODE_NORMAL and self.cursor.visible and self.useFilterMean):
            visible = True
        for obj in self.renderListMean:
            obj.visible = visible
            
        # determine visibility of time points
        visible = False
        if(self.mode == MODE_NORMAL and self.cursor.visible and self.useFilterTime):
            visible = True
        for obj in self.renderListTime:
            obj.visible = visible
        
        # determine visibility of grid array
        visible = False
        if(self.mode == MODE_NORMAL and self.useFilterGrid):
            visible = True
        for obj in self.renderListGrid:
            obj.visible = visible
                
    # handle the key inputs    
    def handleKeys(self, key):
        if(key == ord(KEY_HELP)):
            if(self.mode == MODE_NORMAL):
                self.removeRenderObjects()
                self.showCommandList()
        elif(key == ord(KEY_QUIT)):
            return 0
        elif(key == ord(KEY_CALIBRATE)):
            if(self.mode == MODE_NORMAL):
                self.calibrationDataBackup = self.calibrationData.copy()
                self.mode = MODE_CAMERA_SIZE
                self.detector.trainThreshold()
                self.removeRenderObjects()
                self.pushText('Try to adjust your location and lightning, until the eyes are detected well.', 100)
                self.pushText('Push <' + KEY_INC + '> or <' + KEY_DEC + '> to change the image size.', 100)
                self.pushText('Push <' + KEY_NEXT + '> when you are finished.', 100)
            else:
                self.calibrationData = self.calibrationDataBackup.copy()
                self.mode = MODE_NORMAL
                self.removeRenderObjects()
                self.pushText('The calibration was aborted!', 100)
        elif(key == ord(KEY_RESET)):
            self.mode = MODE_NORMAL
            self.calibrationData = []
            self.calibrated = False
            self.removeRenderObjects()
            self.pushText('Deleted all calibration data.', 100)
        elif(key == ord(KEY_MEAN)):
            self.useFilterMean = not self.useFilterMean
        elif(key == ord(KEY_TIME)):
            self.useFilterTime = not self.useFilterTime
        elif(key == ord(KEY_GRID)):
            self.useFilterGrid = not self.useFilterGrid
        elif(key == ord(KEY_NEXT)):
            if(self.mode == MODE_CAMERA_SIZE):
                self.mode = MODE_THRESHOLD
                self.removeRenderObjects()
                self.pushText('Try to get mainly the pupil detected.', 100)
                self.pushText('Push <' + KEY_INC + '> or <' + KEY_DEC + '> to change the threshold value.', 100)
                self.pushText('Push <n> when you are finished.', 100)
            elif(self.mode == MODE_THRESHOLD):
                self.mode = MODE_CALIBRATE_CURSOR
                self.cursorLocation = 0
                self.removeRenderObjects()
                self.pushText('Follow the orange circle with your eyes.', 100)
                self.pushText('Push <' + KEY_NEXT + '> when you are looking at the circle.', 100)
            elif(self.mode == MODE_CALIBRATE_CURSOR):
                self.calibrationData.append((self.cursor.x, self.cursor.y, np.copy(self.values)))
                self.cursorLocation += 1
                if(self.cursorLocation == len(self.locations)):
                    self.mode = MODE_NORMAL
                    self.computeCalibration()
                    self.calibrated = True
                    self.removeRenderObjects()
                    self.pushText('Calibration finished!', 100)
        elif(key == ord(KEY_INC)):
            if(self.mode == MODE_CAMERA_SIZE):
                if(self.outputScale < 2.0):
                    self.outputScale += 0.1
            elif(self.mode == MODE_THRESHOLD):
                if(self.detector.threshold < 250):
                    self.detector.threshold += 1
        elif(key == ord(KEY_DEC)):
            if(self.mode == MODE_CAMERA_SIZE):
                if(self.outputScale > 0.2):
                    self.outputScale -= 0.1
            elif(self.mode == MODE_THRESHOLD):
                if(self.detector.threshold > 2):
                    self.detector.threshold -= 1
        elif(key == ord(KEY_SAVE)):
            if(self.mode == MODE_NORMAL):
                self.removeRenderObjects()
                if(self.calibrated):
                    with open('calibrationData.dat', 'wb') as fp:
                        pickle.dump(self.calibrationData, fp)
                    self.pushText('Saved calibration data.', 100)
                else:
                    self.pushText('Error! Not calibrated.', 100)
        elif(key == ord(KEY_LOAD)):
            if(self.mode == MODE_NORMAL):
                self.removeRenderObjects()
                if(os.path.exists('calibrationData.dat')):
                    with open('calibrationData.dat', 'rb') as fp:
                        self.calibrationData = pickle.load(fp)
                        self.computeCalibration()
                        self.calibrated = True
                        self.pushText('Loaded calibration data.', 100)
                else:
                    self.pushText('Error! No calibration data available.', 100)
        elif(key == ord(KEY_ROWS_INC)):
            if(self.gridRows < GRID_MAX):
                self.gridRows += 1
                self.createRenderListGrid()
        elif(key == ord(KEY_ROWS_DEC)):
            if(self.gridRows > GRID_MIN):
                self.gridRows -= 1
                self.createRenderListGrid()
        elif(key == ord(KEY_COLS_INC)):
            if(self.gridCols < GRID_MAX):
                self.gridCols += 1
                self.createRenderListGrid()
        elif(key == ord(KEY_COLS_DEC)):
            if(self.gridCols > GRID_MIN):
                self.gridCols -= 1
                self.createRenderListGrid()
        elif(key == ord(KEY_USE_HEAD)):
            self.useHead = not self.useHead
            self.computeCalibration()
        return 1
    
    def run(self):
        # create gui window in fullscreen mode
        cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # create capture device
        videoCapture = cv2.VideoCapture(DEVICE_ID)

        # main loop
        while(True):            
            # save current time at the beginning of each loop
            startTime = time.time()
            
            # try to read an image
            returnValue, self.inputImage = videoCapture.read()
            if(returnValue == 0): # no image could be read
                self.pushText('No webcam found!', 50)
                self.mode = MODE_NORMAL
                self.calibrationData = []
                self.calibrated =  False
            else: # feed the image to the faceDetector and save the values
                self.detector.detect(self.inputImage)
                self.values = self.detector.values
                self.lost = self.detector.lost
            
            self.determineVisibility()
            
            # determine image of outputImage
            width = self.inputImage.shape[1]
            height = self.inputImage.shape[0]
            self.outputImage.width = int(round(self.outputScale*width))
            self.outputImage.height = int(round(self.outputScale*height))
            if(self.mode == MODE_CAMERA_SIZE):
                self.outputImage.image = self.detector.getMarkedImage()
            elif(self.mode == MODE_THRESHOLD):
                self.outputImage.image = self.detector.getBinaryEyeImage(True)
            
            # determine location of cursor
            self.locateCursor()
            
            # render and show current image
            img = self.renderImage()
            cv2.imshow(self.windowName, img)
            
            # read and handle pressed keys
            key = cv2.waitKey(1)
            if(not self.handleKeys(key)):
                break
            
            # calculate FPS
            timeDiff = time.time() - startTime
            self.FPS = int(round(1/timeDiff))
        cv2.destroyAllWindows()
        

eyetracking = EyeTracking()
eyetracking.run()