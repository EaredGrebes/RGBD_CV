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

greyMat = 0.299*rgbMat[:,:,0] + 0.587*rgbMat[:,:,1] + 0.114*rgbMat[:,:,2]
greyMat = greyMat.astype(int)
greyBlurMat = myCv.blurMat(greyMat, maskMat).astype(dtype = np.ubyte)

# blur RGB
rgbBlurMat = np.zeros((height, width, 3), dtype = np.ubyte)
for clr in range(3):
    rgbBlurMat[:,:,clr] = myCv.blurMat(rgbMat[:,:,clr], maskMat).astype(dtype = np.ubyte)
    

# compute rgb entropy on a course grid
# l = 16
# coarseGridHMat, coarseGridHMaskMat = myCv.coarseGridEntropy(rgbBlurMat[:,:,0], rgbBlurMat[:,:,1], rgbBlurMat[:,:,2], maskMat, l)
# HnormMat = cvFun.normalizeEntropymat(coarseGridHMat, coarseGridHMaskMat, -4, 4)

Hmat, Hmask = myCv.computeWindowedImageEntropy(rgbBlurMat, maskMat)
HnormMat = cvFun.normalizeEntropymat(Hmat, Hmask, -4, 4)

HdiffMat = -myCv.LoGMat(HnormMat, Hmask)
HdiffMat = cvFun.normalizeEntropymat(HdiffMat, Hmask, -4, 4)

# entropy threshold
# Hthresh = 0.6 # lower means more points accepted
# coarseGridHMaskMat = HnormMat > Hthresh

# rgbH = np.copy(rgbBlurMat)
# rgbH[np.invert(maskHGrid)] = 0

# get frame 2 mats
print('frame 2 mats')
frame2 = frame1 + 1
rgb2Mat, xyz2Mat, mask2Mat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)

grey2Mat = 0.299*rgb2Mat[:,:,0] + 0.587*rgb2Mat[:,:,1] + 0.114*rgb2Mat[:,:,2]
grey2Mat = grey2Mat.astype(int)
grey2BlurMat = myCv.blurMat(grey2Mat, maskMat).astype(dtype = np.ubyte)

# blur RGB
rgb2BlurMat = np.zeros((height, width, 3), dtype = np.ubyte)
for clr in range(3):
    rgb2BlurMat[:,:,clr] = myCv.blurMat(rgb2Mat[:,:,clr], mask2Mat).astype(dtype = np.ubyte)
   


#------------------------------------------------------------------------------
# some plotting
plt.close('all')

plt.figure('mask')
plt.title('mask')
plt.imshow(maskMat)

plt.figure('rgb frame' + str(frame1))
plt.title('rgb frame ' + str(frame1))
plt.imshow(rgbBlurMat)

plt.figure('grey frame' + str(frame1))
plt.title('grey frame ' + str(frame1))
plt.imshow(greyBlurMat)

plt.figure('rgb frame' + str(frame2))
plt.title('rgb frame ' + str(frame2))
plt.imshow(rgb2BlurMat)

plt.figure('rgb entropy')
plt.title('rgb entropy')
plt.imshow(HnormMat)

plt.figure('H diff')
plt.title('H diff')
plt.imshow(HdiffMat)

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

