import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import sys
import open3d as o3d

 # custom functions
sys.path.insert(1, '../functions')
import video_functions as vid
#import feature_detection_functions as fd
import feature_detection_functions as fd
import feature_detection_functions_gpu as fdgpu
import cv_functions as cvFun

loadData = False
runCPU = False
 
#------------------------------------------------------------------------------
# data configuration and loading

if loadData:
    # where's the data?
    folder = '../../data/'
    
    # calibration data
    numpyName = folder + 'rawData.npz'
    calName = folder + 'calibration.h5'
        
    start = time.time()
    redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(calName, numpyName, folder)
    print('timer:', time.time() - start)
   

#------------------------------------------------------------------------------
# get frames

(height, width, nFrms) = redTens.shape
print('number of frames: ', nFrms)

# create object for handling CV functions
myCv = cvFun.myCv(height, width) 

# get frame 1 mats
print('frame 1 mats')
frame1 = 60
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
greyMat = cvFun.rgbToGreyMat(rgbMat).astype(int) 
height, width = maskMat.shape

#------------------------------------------------------------------------------
# CPU implementation testing

if runCPU:
    c = 8
    cornerObj = fd.corner_detector_class()
    gradxMat, gradyMat = fd.computeGradientMat(cornerObj.pixelOffsetMat, cornerObj.B, greyMat, maskMat, height, width)
    
    crossProdMat = fd.computeCrossProdMat(gradxMat, gradyMat, height, width)

    coarseMaxMat = fd.findLocalMax(crossProdMat, c, height, width)
    
    
#------------------------------------------------------------------------------
# GPU implementation testing
c = 8
nFeatures = 128

height_c = int(height / c)
width_c = int(width / c)

cornerObj = fdgpu.corner_detector_class(height, width, c, nFeatures)

# inputs
greyMat_gpu = cp.array(greyMat, dtype = cp.float32)
maskMat_gpu = cp.array(maskMat, dtype = bool)

# outputs
cornerPointIdx_gpu = cp.zeros((2,nFeatures), dtype = cp.int32)

start = time.time()
cornerPointIdx = cornerObj.findCornerPoints(cornerPointIdx_gpu, greyMat_gpu, maskMat_gpu)
print('timer:', time.time() - start)

# to cpu
cornerPointIdx = cornerPointIdx_gpu.get()

#------------------------------------------------------------------------------
# verification
plt.close('all')

Nmatches = cornerPointIdx.shape[1]
rgb1Match = np.copy(rgbMat)

for ii in range(Nmatches):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerPointIdx[0, ii], cornerPointIdx[1, ii], 8, color.astype(np.ubyte))
    
plt.figure()
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)







