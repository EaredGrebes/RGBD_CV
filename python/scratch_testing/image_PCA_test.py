plotOpen3d = False
plotRgbHist = False

# packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import multiprocessing as mp
import open3d as o3d
import functools as ft
import cv_functions as cvFun

# custom functions
import video_functions as vid
import plot_functions as vdp

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
redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(videoDat, vdid, calName, numpyName)
print('timer:', time.time() - start)
   
#------------------------------------------------------------------------------
# do some processing
(height, width, nFrms) = redTens.shape
print('number of frames: ', nFrms)

# create object for handling CV functions
myCv = cvFun.myCv(height, width) 

# get frame 1 mats
print('frame 1 mats')
frame1 = 15
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)
depthMat = -xyzMat[:,:,2]

# prototype a PCA algorithm for segmentation
xyz, rgb = vid.genPointCloud(xyzMat, rgbMat, maskMat)
depthPoints = -xyz[:,2]
redPoints = rgb[:,0]
greenPoints = rgb[:,1]
bluePoints = rgb[:,2]

# a set of gaussian basis functions for color and depth, whose means are chosen as the mean
# color and depth selected from a uniformly sampled grid of the 2d image
Ngrid = 16
xStep = int(height/Ngrid)
yStep = int(width/Ngrid)

depthMeans = np.zeros(Ngrid*Ngrid)
redMeans     = np.zeros(Ngrid*Ngrid)
greenMeans     = np.zeros(Ngrid*Ngrid)
blueMeans     = np.zeros(Ngrid*Ngrid)

for ii in range(Ngrid):
    for jj in range(Ngrid):
        
        depthRect = depthMat[ii*xStep:ii*xStep + xStep, jj*yStep:jj*yStep + yStep]
        redRect   =   rgbMat[ii*xStep:ii*xStep + xStep, jj*yStep:jj*yStep + yStep, 0]
        greenRect =   rgbMat[ii*xStep:ii*xStep + xStep, jj*yStep:jj*yStep + yStep, 1]
        blueRect   =  rgbMat[ii*xStep:ii*xStep + xStep, jj*yStep:jj*yStep + yStep, 2]
        maskRect =   maskMat[ii*xStep:ii*xStep + xStep, jj*yStep:jj*yStep + yStep]
        
        depthMeans[jj*Ngrid + ii] = np.mean(depthRect[maskRect])
        redMeans[jj*Ngrid + ii] = np.mean(redRect[maskRect])
        greenMeans[jj*Ngrid + ii] = np.mean(greenRect[maskRect])
        blueMeans[jj*Ngrid + ii] = np.mean(blueRect[maskRect])

depthMeans = np.linspace(depthPoints.min(), depthPoints.max(), Ngrid)
redMeans = np.linspace(rgb[:,0].min(), rgb[:,0].max(), Ngrid)
greenMeans = np.linspace(rgb[:,1].min(), rgb[:,1].max(), Ngrid)
blueMeans = np.linspace(rgb[:,2].min(), rgb[:,2].max(), Ngrid)

# now compute the feature matrix
def gaussianKernel(distances, scale):
    return np.exp(-np.power(distances/scale,2))

depthScale = 50
depthDist = np.abs(np.subtract.outer(depthMeans, depthPoints)) 
FeatureDepthMat = gaussianKernel(depthDist, depthScale)

colorScale = 30
dist = np.abs(np.subtract.outer(redMeans, redPoints)) 
FeatureRedMat = gaussianKernel(dist, colorScale)

dist = np.abs(np.subtract.outer(greenMeans, greenPoints)) 
FeatureGreenMat = gaussianKernel(dist, colorScale)

dist = np.abs(np.subtract.outer(blueMeans, bluePoints)) 
FeatureBlueMat = gaussianKernel(dist, colorScale)

FeatureMat = np.concatenate((FeatureDepthMat, FeatureRedMat, FeatureGreenMat, FeatureBlueMat), axis = 0)
#FeatureMat = np.concatenate((rgb, depthPoints[:,None]), axis = 1).T
#FeatureMat = np.concatenate((FeatureRedMat, FeatureGreenMat, FeatureBlueMat), axis = 0)
#FeatureMat = FeatureDepthMat

mn = np.mean(FeatureMat, axis = 1)
std = np.std(FeatureMat, axis = 1)
FeatureMat = (FeatureMat - mn[:,None]) 

covMat = FeatureMat @ FeatureMat.T
U, S, VT = np.linalg.svd(covMat)

Xproj = U.T @ FeatureMat

# mode1 = U[:,4].reshape(Ngrid,Ngrid)
# plt.figure('principal modes')
# plt.title('principal modes')
# plt.imshow(mode1)

clrs = np.concatenate( (rgb/255, np.ones(len(rgb))[:,None]), axis = 1 )


tmp = Xproj[0:3,:]
#tmp = Xproj[[2,3,4],:]
pcd_pca = o3d.geometry.PointCloud()
pcd_pca.points = o3d.utility.Vector3dVector(tmp.T)
pcd_pca.colors = o3d.utility.Vector3dVector(rgb / 255)  # open3d expects color between [0, 1]
o3d.visualization.draw_geometries([pcd_pca])
    
#------------------------------------------------------------------------------
# some plotting
plt.close('all')


plt.figure('projection')
plt.title('projection')
plt.plot(tmp[0,:], tmp[1,:], '.')


plt.figure('singular values of depth')
plt.title('singular values of depth')
plt.plot(S)

plt.figure('mask')
plt.title('mask')
plt.imshow(maskMat)

plt.figure('rgb frame' + str(frame1))
plt.title('rgb frame ' + str(frame1))
plt.imshow(rgbMat)

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].set_title('depth histogram')
axs[0].hist(-xyz[:,2], density=True, bins=128)

axs[1].set_title('depth basis means')
axs[1].plot(depthMeans, 0*depthMeans, 'o')

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].set_title('red histogram')
axs[0].hist(rgb[:,0], density=True, bins=128)

axs[1].set_title('red basis means')
axs[1].plot(redMeans, 0*redMeans, 'o')

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].set_title('green histogram')
axs[0].hist(rgb[:,1], density=True, bins=128)

axs[1].set_title('green basis means')
axs[1].plot(greenMeans, 0*greenMeans, 'o')

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].set_title('blue histogram')
axs[0].hist(rgb[:,2], density=True, bins=128)

axs[1].set_title('blue basis means')
axs[1].plot(blueMeans, 0*blueMeans, 'o')


if plotOpen3d:
    pcd, xyz, rgb = vid.genOpen3dPointCloud(xyzMat, rgbMat, maskMat)
    o3d.visualization.draw_geometries([pcd])

if plotRgbHist:
    # compute joint pdf of RGB data
    pcd, xyz, rgb = vid.genPointCloud(xyzMat, rgbMat)
    rgbPdf = myCv.estimateRgbPdf(rgbMat[:,:,0], rgbMat[:,:,1], rgbMat[:,:,2], maskMat)
    vdp.plotRgbHistogram(rgb*255, rgbPdf)
    

