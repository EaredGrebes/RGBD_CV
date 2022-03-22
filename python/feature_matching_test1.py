import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import open3d as o3d

 # custom functions
sys.path.insert(1, 'functions')
import video_functions as vid
#import feature_detection_functions as fd
import feature_detection_functions_gpu as fd
import cv_functions as cvFun

loadData = True
 
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
frame1 = 24
rgb1Mat, xyz1Mat, mask1Mat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
grey1Mat = cvFun.rgbToGreyMat(rgb1Mat).astype(int) 

print('frame 2 mats')
frame2 = frame1 + 1
rgb2Mat, xyz2Mat, mask2Mat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)
grey2Mat = cvFun.rgbToGreyMat(rgb2Mat).astype(int) 

 
#------------------------------------------------------------------------------
# Testing
nMax = 150
cornerObj = fd.corner_detector_class()

print('corner frame 1 detection')  
corner1Mask, coarseGradMat, cornerIdx1, gradMat = cornerObj.findCornerPoints(grey1Mat, mask1Mat, nMax)

print('corner frame 2 detection')
corner2Mask, coarseGradMat, cornerIdx2, grad2Mat = cornerObj.findCornerPoints(grey2Mat, mask2Mat, nMax)

print('corner matching')
costMatrix = np.zeros((nMax, nMax))
for id1 in range(nMax):
    for id2 in range(nMax):
        costMatrix[id1, id2] = fd.computeMatchCost(rgb1Mat, rgb2Mat, mask1Mat, mask2Mat, cornerIdx1[:,id1], cornerIdx2[:,id2]) 

# find a simpler way of doing this
colMin = costMatrix.argmin(axis = 0)
rowMin = costMatrix.argmin(axis = 1)
fcolMin = costMatrix.min(axis = 0)

match1Mat = np.zeros((nMax, nMax))
match2Mat = np.zeros((nMax, nMax))

match1Mat[colMin, np.arange(nMax)] = 1
match2Mat[np.arange(nMax), rowMin] = 1

matchMat = (match1Mat==1) & (match2Mat ==1)
meanMatchCost = costMatrix[matchMat].mean()

idxMatch = (np.arange(1, nMax+1) @ matchMat) - 1

idxIm2 = np.arange(nMax)[idxMatch >= 0]
idxIm1 = idxMatch[idxMatch >= 0]
  
cornerMatchedIdx1 = cornerIdx1[:, idxIm1]
cornerMatchedIdx2 = cornerIdx2[:, idxIm2]

# 3-d frame registration
print('3-d transformation')

l = 3 # create a box of (2l+1) around each center point
nBox = (2*l + 1) * (2*l + 1)
nMatchedPoints = cornerMatchedIdx1.shape[1]
print('Number of matched points: {}'.format(nMatchedPoints))

xyzPoints1 = np.zeros((nBox * nMatchedPoints, 3))
xyzPoints2 = np.zeros((nBox * nMatchedPoints, 3))

rgbPoints1 = np.zeros((nBox * nMatchedPoints, 3))
rgbPoints2 = np.zeros((nBox * nMatchedPoints, 3))

rgbPoints1[:,0] = 255
rgbPoints1[:,1] = 0
rgbPoints1[:,2] = 0

rgbPoints2[:,0] = 0
rgbPoints2[:,1] = 255
rgbPoints2[:,2] = 0

def computeVectorDistances(xyzPoints1, xyzPoints2):
    
    xyzError = xyzPoints1 - xyzPoints2
    xyzDist = np.sqrt(np.sum(xyzError*xyzError, axis = 1))
    return xyzDist


def addPoints(points, mat, center_x, center_y, l, nBox):
    points[ii*nBox:(ii+1)*nBox, 0] = mat[center_x-l:center_x+l+1, center_y-l:center_y+l+1, 0].flatten()
    points[ii*nBox:(ii+1)*nBox, 1] = mat[center_x-l:center_x+l+1, center_y-l:center_y+l+1, 1].flatten()
    points[ii*nBox:(ii+1)*nBox, 2] = mat[center_x-l:center_x+l+1, center_y-l:center_y+l+1, 2].flatten()
    
def solveTransform(xyzPoints1, xyzPoints2):
    
    Npoints = xyzPoints1.shape[0]
    mean1 = xyzPoints1.mean(axis = 0)
    mean2 = xyzPoints2.mean(axis = 0)

    xyzPoints1_c = xyzPoints1 - mean1[None,:]
    xyzPoints2_c = xyzPoints2 - mean2[None,:]

    # cross covariance matrix
    C = (1/pointMask.sum()) * xyzPoints1_c.T @ xyzPoints2_c
    U, S, VT = np.linalg.svd(C, full_matrices=True)

    R = VT.T @ U.T
    t = mean2 - R @ mean1

    # apply transform
    xyzPoints1_2 = t[None,:].T + R @ xyzPoints1.T
    xyzPoints1_2 = xyzPoints1_2.T
    
    distanceErrors = computeVectorDistances(xyzPoints1_2, xyzPoints2)
    
    return R, t, distanceErrors, xyzPoints1_2

    
for ii in range(nMatchedPoints):
    
    center1_x = cornerMatchedIdx1[0,ii]
    center1_y = cornerMatchedIdx1[1,ii]
    addPoints(xyzPoints1, xyz1Mat, center1_x, center1_y, l, nBox)
    #addPoints(rgbPoints1, rgb1Mat, center1_x, center1_y, l, nBox)
    
    center2_x = cornerMatchedIdx2[0,ii]
    center2_y = cornerMatchedIdx2[1,ii]
    addPoints(xyzPoints2, xyz2Mat, center2_x, center2_y, l, nBox)
    #addPoints(rgbPoints2, rgb2Mat, center2_x, center2_y, l, nBox)
 
pointMask = (abs(xyzPoints1[:,2]) > 0) & (abs(xyzPoints2[:,2]) > 0)

xyzPoints1 = xyzPoints1[pointMask,:]
xyzPoints2 = xyzPoints2[pointMask,:]

rgbPoints1 = rgbPoints1[pointMask,:]
rgbPoints2 = rgbPoints2[pointMask,:]

distErr1 = computeVectorDistances(xyzPoints1, xyzPoints2)
R, t, distErr2, xyzPoints1_2 = solveTransform(xyzPoints1, xyzPoints2)

Npoints = xyzPoints2.shape[0]
idxSmallestDist = np.argsort(distErr2)[:int(Npoints/2)]

R, t, distErr2, xyzPoints1_2 = solveTransform(xyzPoints1[idxSmallestDist,:], xyzPoints2[idxSmallestDist,:])


#------------------------------------------------------------------------------         
# visualize matches
plt.close('all')

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(xyzPoints1_2)
pcd1.colors = o3d.utility.Vector3dVector(rgbPoints1[idxSmallestDist,:] / 255)  # open3d expects color between [0, 1]

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(xyzPoints2[idxSmallestDist,:])
pcd2.colors = o3d.utility.Vector3dVector(rgbPoints2[idxSmallestDist,:] / 255)  # open3d expects color between [0, 1]

o3d.visualization.draw_geometries([pcd1, pcd2])

Nmatches = cornerMatchedIdx1.shape[1]
rgb1Match = np.copy(rgb1Mat)
rgb2Match = np.copy(rgb2Mat)
for ii in range(Nmatches):
    
    c = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchedIdx1[0, ii], cornerMatchedIdx1[1, ii], 9, c.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchedIdx2[0, ii], cornerMatchedIdx2[1, ii], 9, c.astype(np.ubyte))
    
plt.figure('rgb  match' + str(frame1))
plt.title('rgb match ' + str(frame1))
plt.imshow(rgb1Match)

plt.figure('rgb  match' + str(frame2))
plt.title('rgb match ' + str(frame2))
plt.imshow(rgb2Match)


plt.figure('costMatrix')
plt.title('costMatrix')
plt.hist(costMatrix.flatten(), density=True, bins = 300)

plt.figure('fmin cost')
plt.hist(fcolMin, density=True)

plt.figure('coarseGradMat hist')
plt.title('coarseGradMat hist')
plt.hist(gradMat.flatten(), density=True, bins = 300)

plt.figure('distErr')
plt.title('distErr')
plt.hist(distErr1, density=True, bins = 200)
plt.hist(distErr2, density=True, bins = 200)