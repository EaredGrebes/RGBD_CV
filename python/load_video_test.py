import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import video_functions as vid
import time
import h5py
import multiprocessing as mp

print(sys.executable)
print("Number of cpu: ", mp.cpu_count())

loadCal = True
loadVidPar = True

#------------------------------------------------------------------------------
width = 848
height = 480

calName = '../data/calibration.h5'
vidNames = ['../data/videoCaptureTest1.avi', \
            '../data/videoCaptureTest2.avi', \
            '../data/videoCaptureTest3.avi'  ]
    
if loadCal:
    datH5 = h5py.File(calName)
    print(datH5.keys())
    pixelXPosMat = np.array(datH5["pixelXPosMat"][:])
    pixelYPosMat = np.array(datH5["pixelYPosMat"][:])  
    
if loadVidPar:
    start = time.time()
    vidTensorList = vid.loadVideos(vidNames, width, height)
    print('timer:', time.time() - start)
    

#------------------------------------------------------------------------------
# some plotting
for ii in range(len(vidNames)):
    plt.figure()
    plt.spy(vidTensorList[ii][:,:,4])

