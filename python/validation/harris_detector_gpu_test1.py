import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import sys
import open3d as o3d

 # custom functions
sys.path.insert(1, '../functions')
import video_functions as vid
import harris_detector_functions_gpu as hd
import feature_detection_functions as fd

loadData = True

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

# get frame 1 mats
print('frame 1 mats')
frame1 = 40
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)

    
#------------------------------------------------------------------------------
# GPU implementation testing

#  harris detector
nFeatures = 128
hdObj = hd.HarrisDetectorGpu(height, width, 8, nFeatures)

# inputs
greyMat = hd.rgb_to_grey(rgbMat.astype(float))

greyMat_gpu = cp.array(greyMat, dtype = cp.float32)
maskMat_gpu = cp.array(maskMat, dtype = cp.float32)

# working variables
greyBlurMat_gpu = cp.zeros((height, width), dtype = cp.float32)
xgradMat_gpu = cp.zeros((height, width), dtype = cp.float32)
ygradMat_gpu = cp.zeros((height, width), dtype = cp.float32)

# outputs
cornerPointIdx_gpu = cp.zeros((2,nFeatures), dtype = cp.int32)

start = time.time()
hdObj.findCornerPoints(cornerPointIdx_gpu, greyMat_gpu, maskMat_gpu)
cornerPointIdx = cornerPointIdx_gpu.get()
print('timer:', time.time() - start)
    

greyBlur = hdObj.blurMat.get()
corner = hdObj.cornerMaxMat.get()
# xgradMat = xgradMat_gpu.get()
# ygradMat = ygradMat_gpu.get()

corner_mean = 0
corner2 = np.zeros((height, width))
corner2[corner > corner_mean] = 1
corner2[corner <= corner_mean] = 0

#------------------------------------------------------------------------------
# verification
#plt.close('all')

plt.figure('grey raw')
plt.title('grey raw')
plt.imshow(greyMat)

plt.figure('grey blur')
plt.title('grey blur')
plt.imshow(greyBlur)

plt.figure('corner')
plt.title('corner')
plt.spy(corner2)

# plt.figure('ygradMat')
# plt.title('ygradMat')
# plt.imshow(ygradMat)

plt.figure()
plt.spy(maskMat)

plt.figure()
plt.spy(maskMat_gpu.get())

Nmatches = cornerPointIdx.shape[1]
rgb1Match = np.copy(rgbMat)

for ii in range(Nmatches):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerPointIdx[0, ii], cornerPointIdx[1, ii], 8, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)





