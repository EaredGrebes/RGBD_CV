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
import cv_functions

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
(height, width, nFrms) = vidTensorList[vdid['depth8L']].shape
print('number of frames: ', nFrms)

# create object for handling CV functions
myCv = cv_functions.myCv(height, width) 

# get frame mats
frame = 0
redMat, greenMat, blueMat, depth8LMat, depth8UMat = vid.getFrameMats(vidTensorList, vdid, frame)
xMat, yMat, zMat = vid.genXYZMats(depth8LMat, depth8UMat, pixelXPosMat, pixelYPosMat)  
   
# masking 
depthMaskMat = myCv.computeMaskMat(zMat)
redMaskMat = myCv.computeMaskMat(redMat)
greenMaskMat = myCv.computeMaskMat(greenMat)
blueMaskMat = myCv.computeMaskMat(blueMat)

colorMaskMat = redMaskMat + greenMaskMat + blueMaskMat
maskMat = depthMaskMat * colorMaskMat
maskMat[maskMat > 1] = 1

maskMatB = np.array(maskMat, dtype=bool)

# blur RGB
redMat[maskMatB==False] = 0
greenMat[maskMatB==False] = 0
blueMat[maskMatB==False] = 0
    
redBlur   = myCv.blurMat(redMat,   maskMat).astype(dtype = np.ubyte)
greenBlur = myCv.blurMat(greenMat, maskMat).astype(dtype = np.ubyte)
blueBlur  = myCv.blurMat(blueMat,  maskMat).astype(dtype = np.ubyte)
    
rgb = vid.flattenCombine(redBlur, greenBlur, blueBlur)
maskBflat = maskMatB.flatten()
rgb = rgb[maskBflat, :]

# compute joint pdf of RGB data
rgbPdf = myCv.estimateRgbPdf(redBlur, greenBlur, blueBlur, maskMatB)
   
redHist = np.sum(rgbPdf, axis = (1,2))
greenHist = np.sum(rgbPdf, axis = (0,2))
blueHist = np.sum(rgbPdf, axis = (0,1))


#------------------------------------------------------------------------------
# some plotting
plt.close('all')

vdp.plotRgbHistogram(rgb, rgbPdf)

vdp.plotMask(maskMat, 'maskMat')

vdp.plotRgbMats(redMat, greenMat, blueMat, 'RGB frame 1')
vdp.plotRgbMats(redBlur, greenBlur, blueBlur, 'RGB frame 1 blur')

    
if plotOpen3d:
    
    funGetFrame = ft.partial(vid.generateFrameData, vidTensorList, vdid, pixelXPosMat, pixelYPosMat)
    
    frame = 0
    pcd1, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(frame)
    
    frame = nFrms-1
    pcd2, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(frame)
    
    vdp.plotDual_Open3d(pcd1, pcd2)
    
    vdp.plot3d(xyz)

