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
import feature_matching_functions_gpu as fmgpu
import cv_functions as cvFun

loadData = True
runCPU = False
 
#------------------------------------------------------------------------------
# data configuration and loading

if loadData:
    
     # calibration data
    folder = '../../data/'
    numpyName = folder + 'rawData2.npz'
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
frame1 = 25
rgbMat1, xyzMat1, maskMat1 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

frame2 = frame1 + 1
rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

height, width = maskMat1.shape
    
    
#------------------------------------------------------------------------------
# GPU implementation testing
nFeatures = 128
cornerScale = 8
matchScale = 15
matchOffset = np.int32(matchScale/2)
l = 3
xyzScale = 2*l+1
xyzOffset = l

def generateGpuDataDict(xyzMat, rgbMat, maskMat):
    
    gpuDataDict = {
        'xMat': cp.array(xyzMat[:,:,0], dtype = cp.float32),
        'yMat': cp.array(xyzMat[:,:,1], dtype = cp.float32),
        'zMat': cp.array(xyzMat[:,:,2], dtype = cp.float32),   
        'rMat': cp.array(rgbMat[:,:,0], dtype = cp.float32),
        'gMat': cp.array(rgbMat[:,:,1], dtype = cp.float32),
        'bMat': cp.array(rgbMat[:,:,2], dtype = cp.float32),
        'greyMat': None,
        'maskMat': cp.array(maskMat, dtype = bool)    
    }
    gpuDataDict['greyMat'] = fdgpu.rgbToGreyMat(gpuDataDict['rMat'], gpuDataDict['gMat'], gpuDataDict['bMat'])
    return gpuDataDict
    

# inputs
gpuDat1 = generateGpuDataDict(xyzMat1, rgbMat1, maskMat1)
gpuDat2 = generateGpuDataDict(xyzMat2, rgbMat2, maskMat2)

# corner detector object 
cornerObjGpu = fdgpu.corner_detector_class(height, width, cornerScale, nFeatures)

# matching object
matchObjGpu = fmgpu.feature_matching_class(height, width, nFeatures, matchScale)

# corner points
cornerPointIdx1_gpu = cp.zeros((2,nFeatures), dtype = cp.int32)
cornerPointIdx2_gpu = cp.zeros((2,nFeatures), dtype = cp.int32)

# create points of interest matrix
poi_rMat1_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32)
poi_gMat1_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32)
poi_bMat1_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32)
poi_maskMat1_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = bool)

poi_rMat2_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32)
poi_gMat2_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32)
poi_bMat2_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32)
poi_maskMat2_gpu = cp.zeros((nFeatures*matchScale, matchScale), dtype = bool)

# xyz points matrix
xyzVecMat1_gpu = cp.zeros((nFeatures * xyzScale * xyzScale, 3), dtype = cp.float32)
xyzVecMat2_gpu = cp.zeros((nFeatures * xyzScale * xyzScale, 3), dtype = cp.float32)

# grey points matrix
greyVec1_gpu = cp.zeros((nFeatures * xyzScale * xyzScale), dtype = cp.float32)

# previous corner points
cornerObjGpu.findCornerPoints(cornerPointIdx1_gpu, gpuDat1['greyMat'], gpuDat1['maskMat'])

# current corner points
cornerObjGpu.findCornerPoints(cornerPointIdx2_gpu, gpuDat2['greyMat'], gpuDat2['maskMat'])

# construct matrix with reduced number of pixels, around points of interest
fmgpu.generatePointsOfInterestMat(  poi_rMat1_gpu, \
                                    poi_gMat1_gpu, \
                                    poi_bMat1_gpu, \
                                    poi_maskMat1_gpu, \
                                    gpuDat1['rMat'], \
                                    gpuDat1['gMat'], \
                                    gpuDat1['bMat'], \
                                    gpuDat1['maskMat'], \
                                    cornerPointIdx1_gpu,
                                    matchScale,
                                    matchOffset)

fmgpu.generatePointsOfInterestMat(  poi_rMat2_gpu, \
                                    poi_gMat2_gpu, \
                                    poi_bMat2_gpu, \
                                    poi_maskMat2_gpu, \
                                    gpuDat2['rMat'], \
                                    gpuDat2['gMat'], \
                                    gpuDat2['bMat'], \
                                    gpuDat2['maskMat'], \
                                    cornerPointIdx2_gpu,
                                    matchScale,
                                    matchOffset)
 
# match the points of interest    
cornerMatchedIdx1_gpu, \
cornerMatchedIdx2_gpu = matchObjGpu.computeMatches( poi_rMat1_gpu, \
                                                    poi_gMat1_gpu, \
                                                    poi_bMat1_gpu, \
                                                    poi_maskMat1_gpu,    \
                                                    cornerPointIdx1_gpu, \
                                                    poi_rMat2_gpu, \
                                                    poi_gMat2_gpu, \
                                                    poi_bMat2_gpu, \
                                                    poi_maskMat2_gpu, \
                                                    cornerPointIdx2_gpu)
    
fmgpu.generateXYZVecMat(xyzVecMat1_gpu, \
                        greyVec1_gpu,   \
                        gpuDat1['xMat'], \
                        gpuDat1['yMat'], \
                        gpuDat1['zMat'], \
                        gpuDat1['greyMat'], \
                        gpuDat1['maskMat'], \
                        cornerMatchedIdx1_gpu, \
                        xyzScale,
                        xyzOffset)      

xyzVecMat1 = xyzVecMat1_gpu.get()

nMatches = cornerMatchedIdx1_gpu.shape[1]
nBox = xyzScale * xyzScale
nMatchedPoints = nMatches * nBox

xyzVecMat1 = xyzVecMat1[:nMatchedPoints,:]   

    
# verification
cornerMatchIdx1 = cornerMatchedIdx1_gpu.get()
cornerMatchIdx2 = cornerMatchedIdx2_gpu.get()

def addPoints(points, mat, maskMat, center_x, center_y, l, nBox):
    mask = maskMat[center_x-l:center_x+l+1, center_y-l:center_y+l+1].flatten()
    
    points[ii*nBox:(ii+1)*nBox, 0] = mat[center_x-l:center_x+l+1, center_y-l:center_y+l+1, 0].flatten() * mask.astype(float)
    points[ii*nBox:(ii+1)*nBox, 1] = mat[center_x-l:center_x+l+1, center_y-l:center_y+l+1, 1].flatten() * mask.astype(float)
    points[ii*nBox:(ii+1)*nBox, 2] = mat[center_x-l:center_x+l+1, center_y-l:center_y+l+1, 2].flatten() * mask.astype(float)
    
xyzVecMat1_test = np.zeros((nMatchedPoints, 3))

for ii in range(nMatches):
    
    center1_x = cornerMatchIdx1[0,ii]
    center1_y = cornerMatchIdx1[1,ii]
    addPoints(xyzVecMat1_test, xyzMat1, maskMat1, center1_x, center1_y, l, nBox)


#------------------------------------------------------------------------------
# plotting
plt.close('all')

rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(cornerMatchIdx1.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchIdx1[0, ii], cornerMatchIdx1[1, ii], 8, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchIdx2[0, ii], cornerMatchIdx2[1, ii], 8, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)

plt.figure('rgb  frame 2 interest points')
plt.title('rgb  frame 2 interest points')
plt.imshow(rgb2Match)

check1 = np.isclose(xyzVecMat1, xyzVecMat1_test)
print('check 1: {}'.format(check1.min()))








