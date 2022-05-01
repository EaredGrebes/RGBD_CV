plotOpen3d = True
plotRgbHist = True

# packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import open3d as o3d

# custom functions
sys.path.insert(1, 'functions')

import video_functions as vid
import plot_functions as vdp
import cv_functions as cvFun

#------------------------------------------------------------------------------
# data configuration and loading

# where's the data?
folder = '../data/'

# calibration data
calName = folder + 'calibration.h5'
numpyName = folder + 'rawData.npz'

# video streams
vdid = {'blue': 0, 'green': 1, 'red': 2, 'depth8L': 3, 'depth8U': 4}

videoDat = [{'filename': folder + 'videoCaptureTest1.avi', 'channel': 0}, \
            {'filename': folder + 'videoCaptureTest1.avi', 'channel': 1}, \
            {'filename': folder + 'videoCaptureTest1.avi', 'channel': 2}, \
            {'filename': folder + 'videoCaptureTest2.avi', 'channel': 0}, \
            {'filename': folder + 'videoCaptureTest3.avi', 'channel': 0}] 
    
start = time.time()
redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(videoDat, vdid, calName, numpyName)
print('timer:', time.time() - start)
   

#------------------------------------------------------------------------------
# do some processing
(height, width, nFrms) = redTens.shape
myCv = cvFun.myCv(height, width) 
print('number of frames: ', nFrms)

# get frame 1 mats
frame1 = 15
print('frame: ' + str(frame1))
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)


#------------------------------------------------------------------------------
# plotting
plt.close('all')

plt.figure('mask')
plt.title('mask')
plt.imshow(maskMat)

plt.figure('rgb frame' + str(frame1))
plt.title('rgb frame ' + str(frame1))
plt.imshow(rgbMat)

if plotOpen3d:
    pcd, xyz, rgb = vid.genOpen3dPointCloud(xyzMat, rgbMat, maskMat)
    
    #o3d.visualization.draw_geometries([pcd])
    
    vdp.plot_Open3d(pcd)

    
