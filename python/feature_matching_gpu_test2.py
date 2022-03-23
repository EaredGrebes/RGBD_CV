import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import sys
import open3d as o3d

 # custom functions
sys.path.insert(1, 'functions')
import video_functions as vid
#import feature_detection_functions as fd
import feature_detection_functions as fd
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu
import cv_functions as cvFun

loadData = True
runCPU = False
 
#------------------------------------------------------------------------------
# data configuration and loading

if loadData:
    
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
         
    start = time.time()
    redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(videoDat, vdid, calName, numpyName)
    print('timer:', time.time() - start)
    

#------------------------------------------------------------------------------
# get frames

(height, width, nFrms) = redTens.shape
print('number of frames: ', nFrms)

# create object for handling CV functions
myCv = cvFun.myCv(height, width) 

# get frame 1 mats
frame1 = 12
rgbMat1, xyzMat1, maskMat1 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

frame2 = frame1 + 1
rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

height, width = maskMat1.shape
    
    
#------------------------------------------------------------------------------
# GPU implementation testing
cornerScale = 8
matchScale = 8
nMax = 128

# inputs
rgbMat1_gpu = cp.array(rgbMat1, dtype = cp.float32)
maskMat1_gpu = cp.array(maskMat1, dtype = bool)

rgbMat2_gpu = cp.array(rgbMat2, dtype = cp.float32)
maskMat2_gpu = cp.array(maskMat2, dtype = bool)

# corner detector object 
cornerObjGpu = fdgpu.corner_detector_class(height, width, cornerScale, nMax)

# match object
#matchObjGpu = fmgpu.feature_matching_class(imHeight, imgWidth, nMax, matchScale)

# previous corner points
cornerIdx_p = cornerObjGpu.findCornerPoints(fdgpu.rgbToGreyMat(rgbMat1_gpu), maskMat1_gpu)
cornerIdx_1 = cornerIdx_p.get()

# current corner points
cornerIdx_c = cornerObjGpu.findCornerPoints(fdgpu.rgbToGreyMat(rgbMat2_gpu), maskMat2_gpu)
cornerIdx_2 = cornerIdx_c.get()




#------------------------------------------------------------------------------
# verification
plt.close('all')

rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(nMax):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerIdx_1[0, ii], cornerIdx_1[1, ii], 8, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerIdx_2[0, ii], cornerIdx_2[1, ii], 8, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)

plt.figure('rgb  frame 2 interest points')
plt.title('rgb  frame 2 interest points')
plt.imshow(rgb2Match)





