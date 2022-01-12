plotOpen3d = False
plotRgbHist = True

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

# get frame mats
frame = 25
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame)

# blur RGB
print('blurring')
rgbBlurMat = np.zeros((height, width, 3), dtype = np.ubyte)
for clr in range(3):
    rgbBlurMat[:,:,clr] = myCv.blurMat(rgbMat[:,:,clr], maskMat).astype(dtype = np.ubyte)

# compute rgb entropy
Hmat, Hmask = myCv.computeImageEntropy(rgbBlurMat[:,:,0], rgbBlurMat[:,:,1], rgbBlurMat[:,:,2], maskMat)
HnormMat = cvFun.normalizeEntropymat(Hmat, Hmask, -5, 5)
HBlurMat = myCv.blurMat(HnormMat, maskMat)

# entropy threshold
Hthresh = 0.6 # lower means more points accepted
maskH = HBlurMat < Hthresh
l = 8
thresh = np.round(l*l*0.5).astype(int)
# create a coarse grid, each block in the grid is a mask for an interesting block
maskHGridSmall, maskHGridBig = myCv.gridSpace(np.invert(maskH), l, thresh)

rgbH = np.copy(rgbBlurMat)
rgbH[np.invert(maskHGridBig)] = 0

# finite differences
hs, ws = maskHGridSmall.shape
gradMat = np.zeros((hs, ws, 8))


#------------------------------------------------------------------------------
# some plotting
plt.close('all')

plt.figure()
plt.spy(maskHGridSmall)

plt.figure('rgb entropy')
plt.title('rgb entropy')
plt.imshow(HBlurMat)

plt.figure('mask')
plt.title('mask')
plt.imshow(maskMat)

plt.figure('rgb frame' + str(frame))
plt.title('rgb frame ' + str(frame))
plt.imshow(rgbMat)

frame += 1
rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame)
plt.figure('rgb frame' + str(frame))
plt.title('rgb frame ' + str(frame))
plt.imshow(rgbMat)

plt.figure('rgb entropy')
plt.title('rgb entropy')
plt.imshow(HnormMat)

plt.figure('rgb entropy filtered')
plt.title('rgb entropy filtered')
plt.imshow(rgbH)

plt.figure('rgb H hist')
plt.title('rgb H hist')
plt.hist(HnormMat.flatten(), density=True, bins = 300)

if plotOpen3d:
    pcd, xyz, rgb = vid.genPointCloud(xyzMat, rgbMat)
    o3d.visualization.draw_geometries([pcd])

if plotRgbHist:
    # compute joint pdf of RGB data
    pcd, xyz, rgb = vid.genPointCloud(xyzMat, rgbMat)
    rgbPdf = myCv.estimateRgbPdf(rgbBlurMat[:,:,0], rgbBlurMat[:,:,1], rgbBlurMat[:,:,2], maskMat)
    vdp.plotRgbHistogram(rgb*255, rgbPdf)
    
# # plt.figure('xyz H hist')
# # plt.title('xyz H hist')
# # plt.hist(H1mat.flatten(), density=True, bins = 300)

# vdp.plotMask(maskMat, 'maskMat')

# frm = str(frame)
# vdp.plotRgbMats(redMat, greenMat, blueMat, 'RGB frame ' + frm)
# vdp.plotRgbMats(redBlur, greenBlur, blueBlur, 'RGB blur frame ' + frm)
# vdp.plotRgbMats(rM, gM, bM, 'RGB blur mask frame ' + frm)


        
# if plotOpen3d: 
#     funGetFrame = ft.partial(vid.generateFrameData, vidTensorList, vdid, pixelXPosMat, pixelYPosMat)
    
#     pcd, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(frame)
#     vdp.plot_Open3d(pcd)
    
#     #frame = 0
#     #pcd1, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(frame)
#     #frame = nFrms-1
#     #pcd2, xMat, yMat, zMat, redMat, greenMat, blueMat, xyz, rgb = funGetFrame(frame)
#     #vdp.plotDual_Open3d(pcd1, pcd2)
    
#     #vdp.plot3d(xyz)

