import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import video_functions as vid

print(sys.executable)

#------------------------------------------------------------------------------
vidName = '../data/videoCaptureTest2.avi'

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

#------------------------------------------------------------------------------

plt.figure()
plt.spy(frameMat)

