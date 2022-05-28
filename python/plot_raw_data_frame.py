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
numpyName = folder + 'rawData1.npz'
calName = folder + 'calibration.h5'
    
start = time.time()
redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(calName, numpyName, folder)
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

    
