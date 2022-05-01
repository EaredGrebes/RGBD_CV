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
frame1 = 100
rgbMat1, xyzMat1, maskMat1 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

frame2 = frame1 + 40
rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

height, width = maskMat1.shape
    
    
#------------------------------------------------------------------------------
# GPU implementation testing
cornerScale = 8
matchScale = 15
offset = np.int32(1 + matchScale/2)
nFeatures = 128


# inputs
rMat1_gpu = cp.array(rgbMat1[:,:,0], dtype = cp.float32)
gMat1_gpu = cp.array(rgbMat1[:,:,1], dtype = cp.float32)
bMat1_gpu = cp.array(rgbMat1[:,:,2], dtype = cp.float32)
greyMat1_gpu = fdgpu.rgbToGreyMat(rMat1_gpu, gMat1_gpu, bMat1_gpu)
maskMat1_gpu = cp.array(maskMat1, dtype = bool)

rMat2_gpu = cp.array(rgbMat2[:,:,0], dtype = cp.float32)
gMat2_gpu = cp.array(rgbMat2[:,:,1], dtype = cp.float32)
bMat2_gpu = cp.array(rgbMat2[:,:,2], dtype = cp.float32)
greyMat2_gpu = fdgpu.rgbToGreyMat(rMat2_gpu, gMat2_gpu, bMat2_gpu)
maskMat2_gpu = cp.array(maskMat2, dtype = bool)

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

start = time.time()

# previous corner points
cornerObjGpu.findCornerPoints(cornerPointIdx1_gpu, greyMat1_gpu, maskMat1_gpu)

# current corner points
cornerObjGpu.findCornerPoints(cornerPointIdx2_gpu, greyMat2_gpu, maskMat2_gpu)

fmgpu.generatePointsOfInterestMat(  poi_rMat1_gpu, \
                                    poi_gMat1_gpu, \
                                    poi_bMat1_gpu, \
                                    poi_maskMat1_gpu, \
                                    rMat1_gpu, \
                                    gMat1_gpu, \
                                    bMat1_gpu, \
                                    maskMat1_gpu, \
                                    cornerPointIdx1_gpu,
                                    matchScale,
                                    offset)

fmgpu.generatePointsOfInterestMat(  poi_rMat2_gpu, \
                                    poi_gMat2_gpu, \
                                    poi_bMat2_gpu, \
                                    poi_maskMat2_gpu, \
                                    rMat2_gpu, \
                                    gMat2_gpu, \
                                    bMat2_gpu, \
                                    maskMat2_gpu, \
                                    cornerPointIdx2_gpu,
                                    matchScale,
                                    offset)
    
matchObjGpu.createCostMat(poi_rMat1_gpu, \
                          poi_gMat1_gpu, \
                          poi_bMat1_gpu, \
                          poi_maskMat1_gpu, \
                          poi_rMat2_gpu, \
                          poi_gMat2_gpu, \
                          poi_bMat2_gpu, \
                          poi_maskMat2_gpu)

costMat = matchObjGpu.costCoarseMat.get()

colMin = costMat.argmin(axis = 0)
rowMin = costMat.argmin(axis = 1)

idxMatch = rowMin[colMin] == np.arange(nFeatures)

idxMatchImg2 = np.arange(nFeatures)[idxMatch]
idxMatchImg1 = colMin[idxMatchImg2]

cornerMatchedIdx1_gpu = cornerPointIdx1_gpu[:, idxMatchImg1]
cornerMatchedIdx2_gpu = cornerPointIdx2_gpu[:, idxMatchImg2]

print('timer:', time.time() - start)

# verification
nMatchedPoints = cornerMatchedIdx1_gpu.shape[1]
print('Number of matched points: {}'.format(nMatchedPoints))

poi_rMat1 = poi_rMat1_gpu.get()
poi_rMat2 = poi_rMat2_gpu.get()
cornerPointIdx1 = cornerPointIdx1_gpu.get()
cornerPointIdx2 = cornerPointIdx2_gpu.get()
costFineMat = matchObjGpu.costFineMat.get()
costCoarseMat = matchObjGpu.costCoarseMat.get()

cornerMatchIdx1 = cornerMatchedIdx1_gpu.get()
cornerMatchIdx2 = cornerMatchedIdx2_gpu.get()

# slow implementation
offset = np.floor(1 + matchScale/2).astype(int)
costFineMat_test = np.zeros((nFeatures*matchScale, nFeatures*matchScale))   
costCoarseMat_test = np.zeros((nFeatures, nFeatures))

def getPixelRanges(x, y, matchScale, offset):
    x1 = x- offset
    x2 = x + matchScale - offset
    y1 = y - offset
    y2 = y + matchScale - offset  
    
    return x1, x2, y1, y2

def computePoiMat(mat, cornerPointIdx, matchScale, offset, nFeatures):
    
    poiMat_test = np.zeros((nFeatures*matchScale, matchScale))
    for ii in range(nFeatures):
        x1, x2, y1, y2 = getPixelRanges(cornerPointIdx[0,ii] , cornerPointIdx[1,ii] , matchScale, offset)
        poiMat_test[ii*matchScale:(ii+1)*matchScale, :] = mat[x1:x2, y1:y2]
    
    return poiMat_test


Poi_r1_test = computePoiMat(rgbMat1[:,:,0], cornerPointIdx1, matchScale, offset, nFeatures)
Poi_g1_test = computePoiMat(rgbMat1[:,:,1], cornerPointIdx1, matchScale, offset, nFeatures)
Poi_b1_test = computePoiMat(rgbMat1[:,:,2], cornerPointIdx1, matchScale, offset, nFeatures)
Poi_mask1_test = computePoiMat(maskMat1, cornerPointIdx1, matchScale, offset, nFeatures)

Poi_r2_test = computePoiMat(rgbMat2[:,:,0], cornerPointIdx2, matchScale, offset, nFeatures)
Poi_g2_test = computePoiMat(rgbMat2[:,:,1], cornerPointIdx2, matchScale, offset, nFeatures)
Poi_b2_test = computePoiMat(rgbMat2[:,:,2], cornerPointIdx2, matchScale, offset, nFeatures)
Poi_mask2_test = computePoiMat(maskMat2, cornerPointIdx2, matchScale, offset, nFeatures)

c = matchScale    
for ii in range(nFeatures):
    for jj in range(nFeatures):
        
        err1 = Poi_r1_test[ii*c:(ii+1)*c, :] - Poi_r2_test[jj*c:(jj+1)*c, :]
        err2 = Poi_g1_test[ii*c:(ii+1)*c, :] - Poi_g2_test[jj*c:(jj+1)*c, :]
        err3 = Poi_b1_test[ii*c:(ii+1)*c, :] - Poi_b2_test[jj*c:(jj+1)*c, :]
        
        errMat = err1*err1 + err2*err2 + err3*err3
        
        maskMat = np.logical_and(Poi_mask1_test[ii*c:(ii+1)*c, :], Poi_mask2_test[jj*c:(jj+1)*c, :])
        
        errMat[np.invert(maskMat)] = 0

        costFineMat_test[ii*c:(ii+1)*c, jj*c:(jj+1)*c] = errMat
        
        numPoints = np.sum(errMat > 0)
        if (numPoints > 0):
            costCoarseMat_test[ii,jj] = np.sum(errMat) / numPoints


#------------------------------------------------------------------------------
# plotting
plt.close('all')

rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(cornerMatchIdx1.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchIdx1[0, ii], cornerMatchIdx1[1, ii], 12, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchIdx2[0, ii], cornerMatchIdx2[1, ii], 12, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)

plt.figure('rgb  frame 2 interest points')
plt.title('rgb  frame 2 interest points')
plt.imshow(rgb2Match)

check1 = np.isclose(poi_rMat1, Poi_r1_test)
check3 = np.isclose(costFineMat_test, costFineMat)
check4 = np.isclose(costCoarseMat_test, costCoarseMat)

print('check 1: {}'.format(check1.min()))
print('check 3: {}'.format(check3.min()))
print('check 4: {}'.format(check4.min()))

plt.figure('check 3')
plt.spy(check3)

plt.figure('check 4')
plt.spy(check4)





