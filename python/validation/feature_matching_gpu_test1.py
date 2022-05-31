import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import sys
import open3d as o3d

 # custom functions
sys.path.insert(1, '../functions')
import video_functions as vid
import feature_detection_functions as fd
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu
import cv_functions as cvFun

loadData = False
 
#------------------------------------------------------------------------------
# data configuration and loading

if loadData:
    
     # calibration data
    folder = '../../data/'
    calName = folder + 'calibration.h5'
    numpyName = folder + 'rawData.npz'
         
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
frame1 = 220
rgbMat1, xyzMat1, maskMat1 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

frame2 = frame1 + 1
rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)
greyMat1 = cvFun.rgbToGreyMat(rgbMat1).astype(int) 

height, width = maskMat1.shape
    
#------------------------------------------------------------------------------
# GPU implementation testing
cornerScale = 8
matchScale = 13
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

# previous corner points
start = time.time()
cornerObjGpu.findCornerPoints(cornerPointIdx1_gpu, greyMat1_gpu, maskMat1_gpu)

# current corner points
cornerObjGpu.findCornerPoints(cornerPointIdx2_gpu, greyMat2_gpu, maskMat2_gpu)

matchObjGpu.set_img1_features(rMat1_gpu, gMat1_gpu, bMat1_gpu, maskMat1_gpu, cornerPointIdx1_gpu)
matchObjGpu.set_img2_features(rMat2_gpu, gMat2_gpu, bMat2_gpu, maskMat2_gpu, cornerPointIdx2_gpu)
cornerMatchedIdx1, cornerMatchedIdx2 = matchObjGpu.computeFeatureMatches()
print('timer:', time.time() - start) 

cornerMatchIdx1 = cornerMatchedIdx1.get()
cornerMatchIdx2 = cornerMatchedIdx2.get()

print('number of frames matched:')
print(cornerMatchIdx1.shape[1])

#------------------------------------------------------------------------------
# plotting
plt.close('all')

def plot_feature(feat_r, feat_g, feat_b, idx, scale):

    mat = np.stack((np.reshape(feat_r[idx,:], (scale, scale)),
                    np.reshape(feat_g[idx,:], (scale, scale)),
                    np.reshape(feat_b[idx,:], (scale, scale))), axis = 2)
    
    plt.figure()
    plt.imshow(mat/255)

#for idx in range(3):
#    plot_feature(feature_rMat1, feature_gMat1, feature_bMat1, idx, matchScale)   
    
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






