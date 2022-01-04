plotOpen3d = True
loadData = True

if plotOpen3d:
    %reset -f
    plotOpen3d = True
    loadData = True

# packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import multiprocessing as mp
import open3d as o3d
import functools as ft

# custom functions
import video_functions as vid
import plot_functions as vdp

print(sys.executable)
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
vidTensorList, pixelXPosMat, pixelYPosMat, width, height = vid.loadDataSet(videoDat, calName, numpyName)
print('timer:', time.time() - start)
 
            
#------------------------------------------------------------------------------
# do some processing
(h, w, nFrms) = vidTensorList[vdid['depth8L']].shape
print('number of frames: ', nFrms)

funGetFrame = ft.partial(vid.getFrame, vidTensorList, vdid, pixelXPosMat, pixelYPosMat, width, height)

# get point cloud data for the first and last frame
start = time.time()
pcd1, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(0)
pcd2, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(nFrms - 1)
print('timer:', time.time() - start)


#------------------------------------------------------------------------------
# some plotting
plt.close('all')

if plotOpen3d:
    vdp.plot2_Open3d(pcd1, pcd2)
   
else:
    frame = 0
    vdp.plotMask(vidTensorList[vdid['red']][:,:,frame], 'red channel')
    
    vdp.plot3d(xyz)




