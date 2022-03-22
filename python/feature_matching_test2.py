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
import cv_functions as cvFun

loadData = False
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
print('frame 1 mats')
frame1 = 6
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
nMax = 128

height_c = int(height / c)
width_c = int(width / c)

cornerObjGpu = fdgpu.corner_detector_class()

# inputs
greyMat_gpu = cp.array(greyMat, dtype = cp.float32)
maskMat_gpu = cp.array(maskMat, dtype = bool)

# working variables
gradxMat_gpu = cp.zeros((height, width), dtype = cp.float32)
gradyMat_gpu = cp.zeros((height, width), dtype = cp.float32)
crossProdMat_gpu = cp.zeros((height, width), dtype = cp.float32)
coarseMaxMat_gpu = cp.zeros((height, width), dtype = cp.float32)

# outputs
courseMaxVec_gpu = cp.zeros(height_c * width_c, dtype = cp.float32)
pixelXVec_gpu = cp.zeros(height_c * width_c, dtype = cp.int32)
pixelYVec_gpu = cp.zeros(height_c * width_c, dtype = cp.int32)

start = time.time()
fdgpu.findCornerPoints(gradxMat_gpu, \
                     gradyMat_gpu, \
                     crossProdMat_gpu, \
                     coarseMaxMat_gpu, \
                     courseMaxVec_gpu, \
                     pixelXVec_gpu, \
                     pixelYVec_gpu, \
                     greyMat_gpu, \
                     maskMat_gpu, \
                     cornerObjGpu.pixelOffsetMat, 
                     cornerObjGpu.B, 
                     nMax, 
                     c, 
                     height, 
                     width)
print('timer:', time.time() - start)
    
gradxMat_cpu = gradxMat_gpu.get()    
gradyMat_cpu = gradyMat_gpu.get()
crossProdMat_cpu = crossProdMat_gpu.get()
coarseMaxMat_cpu = coarseMaxMat_gpu.get()
courseMaxVec_cpu = courseMaxVec_gpu.get()
pixelXVec_cpu = pixelXVec_gpu.get()
pixelYVec_cpu = pixelYVec_gpu.get()

vec = coarseMaxMat_cpu.flatten()
vecArgSort = vec.argsort()

idxMax = vecArgSort[-nMax:]  # argsort puts the maximum value at the end
idx2dMax = np.array(np.unravel_index(idxMax, (height, width)))

# alternative method
maxVals1 = np.sort(vec)[-nMax:]
maxVals2 = np.sort(courseMaxVec_cpu)[-nMax:]

idxSorted = courseMaxVec_cpu.argsort()[-nMax:]

idx2dMax_2 = np.zeros((2,nMax))

idx2dMax_2[0,:] = pixelXVec_cpu[idxSorted]
idx2dMax_2[1,:] = pixelYVec_cpu[idxSorted]


#------------------------------------------------------------------------------
# verification
plt.close('all')

Nmatches = idx2dMax.shape[1]
rgb1Match = np.copy(rgbMat)

for ii in range(Nmatches):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, idx2dMax[0, ii], idx2dMax[1, ii], 8, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)

plt.figure('corner mask')
plt.title('corner mask')
plt.spy(coarseMaxMat_cpu)

check1 = np.isclose(gradxMat, gradxMat_cpu)
check2 = np.isclose(gradyMat, gradyMat_cpu)
check3 = np.isclose(crossProdMat, crossProdMat_cpu)
check4 = np.isclose(coarseMaxMat, coarseMaxMat_cpu)

plt.figure('check 1')
plt.spy(check1)

plt.figure('check 2')
plt.spy(check2)

plt.figure('check 3')
plt.spy(check3)

plt.figure('check 4')
plt.spy(check4)





