import sys
import matplotlib as plt
import numpy as np
import cv2

print(sys.executable)

#------------------------------------------------------------------------------
vidName = '../data/videoCaptureTest3.avi'

width = 848
height = 480

frameMat = np.array((height, width))
frames = []

cap = cv2.VideoCapture(vidName)
frameCount = 0

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame',frame)
        frameMat = np.asarray( frame[:,:,0] )
        frames.append(frameMat)
        frameCount += 1
    else:
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()