plotOpen3d = False
plotRgbHist = False

# packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import multiprocessing as mp
import open3d as o3d
import functools as ft
import cv_functions as cvFun
import cupy as cp

# custom functions
import video_functions as vid
import plot_functions as vdp

print("Number of cpu: ", mp.cpu_count())
    
#------------------------------------------------------------------------------
# data configuration

# calibration data
folder = '../data/'
calName = folder + 'calibration.h5'
numpyName = folder + 'rawData.npz'

# video streams
vdid = {'blue': 0, 'green': 1, 'red': 2, 'depth8L': 3, 'depth8U': 4}

videoDat = [{'filename': folder + 'videoCaptureTest1.avi', 'channel': 0}, \
            {'filename': folder + 'videoCaptureTest1.avi', 'channel': 1}, \
            {'filename': folder + 'videoCaptureTest1.avi', 'channel': 2}, \
            {'filename': folder + 'videoCaptureTest2.avi', 'channel': 0}, \
            {'filename': folder + 'videoCaptureTest3.avi', 'channel': 0}] 
    
#------------------------------------------------------------------------------
# load data
start = time.time()
redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(videoDat, vdid, calName, numpyName)
print('timer:', time.time() - start)
   
#------------------------------------------------------------------------------
# do some processing
(height, width, nFrms) = redTens.shape
print('number of frames: ', nFrms)

# create object for handling CV functions
myCv = cvFun.myCv(height, width) 

# get frame 1 mats
print('frame 1 mats')
frame1 = 15
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
depthMat = -xyzMat[:,:,2]

# cupy
depthMat_g = cp.array(depthMat)


#------------------------------------------------------------------------------
# some plotting
plt.close('all')

plt.figure('mask')
plt.title('mask')
plt.imshow(maskMat)

plt.figure('rgb frame' + str(frame1))
plt.title('rgb frame ' + str(frame1))
plt.imshow(rgbMat)

if plotOpen3d:
    pcd, xyz, rgb = vid.genOpen3dPointCloud(xyzMat, rgbMat, maskMat)
    o3d.visualization.draw_geometries([pcd])

if plotRgbHist:
    # compute joint pdf of RGB data
    pcd, xyz, rgb = vid.genPointCloud(xyzMat, rgbMat)
    rgbPdf = myCv.estimateRgbPdf(rgbMat[:,:,0], rgbMat[:,:,1], rgbMat[:,:,2], maskMat)
    vdp.plotRgbHistogram(rgb*255, rgbPdf)
    

