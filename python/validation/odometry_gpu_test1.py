import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import sys
import open3d as o3d
from mpl_toolkits import mplot3d

 # custom functions
sys.path.insert(1, '../functions')
import video_functions as vid
#import feature_detection_functions as fd
import feature_detection_functions as fd
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu
import RGBD_odometry_gpu as rgbd
import image_transform_functions_gpu as imgt
import cv_functions as cvFun

plt.close('all')
 
#------------------------------------------------------------------------------
# data configuration and loading
loadData = True
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

height, width = maskMat1.shape
    
#------------------------------------------------------------------------------
# GPU odometry

# initialize some objects
rgbdObj = rgbd.RGBD_odometry_gpu_class(rgbMat1, xyzMat1, maskMat1)

imgTransformObj = imgt.image_transform_class(height, width, rgbdObj.nPoi)

# add new frames
deltaFrames = 20
drVec = np.zeros((nFrms, 3))

for frame in range(frame1, frame1 + deltaFrames):
    print(frame)
    
    rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame)
    
    if frame < frame1 + 1:
        updatePrevious = True
    else:
        updatePrevious = False
            
    optDeltaVec, optNormVec, optCostVec, xyzPoints_p, xyzPoints_c, xyzPoints_pinc = rgbdObj.matchNewFrame(rgbMat2, xyzMat2, maskMat2, updatePrevious)
    
    R_bpTobc = rgbdObj.R_bpTobc
    t_bpTobc_inbc = rgbdObj.t_bpTobc_inbc
    
    drVec[frame,:] = t_bpTobc_inbc
    print(t_bpTobc_inbc)


# some plotting
plt.figure()
plt.plot(drVec)

plt.figure()
plt.plot(optDeltaVec)
# update state

plt.figure()
plt.plot(optNormVec)

plt.figure()
plt.hist(optCostVec[optCostVec != 0], bins = 200)

#------------------------------------------------------------------------------
# verification

#Dcm = cp.array(rgbdObj.R_bpTobc, dtype = cp.float32)
#translation = cp.array(rgbdObj.t_bpTobc_inbc, dtype = cp.float32)

Dcm = cp.array(R_bpTobc, dtype = cp.float32)
translationVec = cp.array(t_bpTobc_inbc, dtype = cp.float32)

# find the feature points in the current frame, using the optimized transform,
# and the point in the previous frame
nMatchedPoints = rgbdObj.cornerMatchedIdx_c.shape[1]
feature_xyzVecMat_p = cp.zeros((nMatchedPoints, 3), dtype = cp.float32)
feature_greyVec_p = cp.zeros((nMatchedPoints), dtype = cp.float32)

xyzScale = 1
xyzOffset = 0

fmgpu.generateXYZVecMat(feature_xyzVecMat_p, \
                        feature_greyVec_p, \
                        rgbdObj.sensorData_p['xMat'], \
                        rgbdObj.sensorData_p['yMat'], \
                        rgbdObj.sensorData_p['zMat'], \
                        rgbdObj.sensorData_p['greyMat'], \
                        rgbdObj.sensorData_p['maskMat'], \
                        rgbdObj.cornerMatchedIdx_p, \
                        xyzScale,
                        xyzOffset)
    
feature_xyzVecMat_c = cp.zeros((nMatchedPoints, 3), dtype = cp.float32)
feature_pixelIdVec = cp.zeros((nMatchedPoints, 2), dtype = cp.int32)
feature_zErrorVec = cp.zeros((nMatchedPoints), dtype = cp.float32)    

imgTransformObj.transformXYZVecMat(feature_xyzVecMat_c, \
                                   feature_pixelIdVec,  \
                                   feature_zErrorVec,   \
                                   feature_xyzVecMat_p, \
                                   rgbdObj.sensorData_c['zMat'],\
                                   Dcm,\
                                   translationVec)     
    
cornerMatchedIdx_p = rgbdObj.cornerMatchedIdx_p.get()  
cornerMatchedIdx_c = rgbdObj.cornerMatchedIdx_c.get()
cornerMatchedIdx_c_opt = feature_pixelIdVec.T.get()    

# tranform all the POIs 
poi_xyzVecMat_c = cp.zeros((rgbdObj.nPoi, 3), dtype = cp.float32)
poi_pixelIdVec = cp.zeros((rgbdObj.nPoi, 2), dtype = cp.int32)
poi_zErrorVec = cp.zeros((rgbdObj.nPoi), dtype = cp.float32)

imgTransformObj.transformXYZVecMat(poi_xyzVecMat_c, \
                                   poi_pixelIdVec,  \
                                   poi_zErrorVec,   \
                                   rgbdObj.xyzVecMat_p,       \
                                   rgbdObj.sensorData_c['zMat'],\
                                   Dcm,\
                                   translationVec)        
 
poi_xyzVecMatTransformed = poi_xyzVecMat_c.get()    
poi_zErrorVec_cpu = poi_zErrorVec.get()
poiMatchedIdx_c_opt = poi_pixelIdVec.T.get()

xyzVecMat3 = np.zeros(poi_xyzVecMatTransformed.shape)

for ii in range(poi_xyzVecMatTransformed.shape[0]):
    
    xyzVecMat3[ii, 0] = xyzMat2[poiMatchedIdx_c_opt[0,ii], poiMatchedIdx_c_opt[1,ii], 0]
    xyzVecMat3[ii, 1] = xyzMat2[poiMatchedIdx_c_opt[0,ii], poiMatchedIdx_c_opt[1,ii], 1]
    xyzVecMat3[ii, 2] = xyzMat2[poiMatchedIdx_c_opt[0,ii], poiMatchedIdx_c_opt[1,ii], 2]

# full depth image transform
xMat_frame1_transformed = cp.zeros((height, width), dtype = cp.float32)
yMat_frame1_transformed = cp.zeros((height, width), dtype = cp.float32)
zMat_frame1_transformed = cp.zeros((height, width), dtype = cp.float32)

zMax = 100000
zMat_frame1_inFrame2 = zMax * cp.ones((height, width), dtype = cp.int32) # has to be int32 for atomic min function in cuda

imgTransformObj.transformDepthImage(xMat_frame1_transformed, \
                                    yMat_frame1_transformed, \
                                    zMat_frame1_transformed, \
                                    zMat_frame1_inFrame2,    \
                                    rgbdObj.sensorData_p['xMat'], \
                                    rgbdObj.sensorData_p['yMat'], \
                                    rgbdObj.sensorData_p['zMat'], \
                                    rgbdObj.sensorData_p['maskMat'], \
                                    Dcm, \
                                    translationVec)

zMat_frame1_inFrame2[zMat_frame1_inFrame2 >= zMax] = 0
test = zMat_frame1_inFrame2.get()

xyzMat_1in2 = np.stack((xMat_frame1_transformed.get(), yMat_frame1_transformed.get(), zMat_frame1_transformed.get()), axis = 2)


#------------------------------------------------------------------------------
# plotting

plt3d = True
if plt3d:
    
    def make3dset(xyzPoints, color):
        rgbPoints = np.tile(color, (xyzPoints.shape[0], 1))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzPoints)
        pcd.colors = o3d.utility.Vector3dVector(rgbPoints)  # open3d expects color between [0, 1]
        return pcd
      
    
    pcd1 = make3dset(xyzPoints_p, np.array([1, 0, 0]))
    pcd2 = make3dset(xyzPoints_c, np.array([0, 1, 1]))
    pcd3 = make3dset(xyzPoints_pinc, np.array([1, 1, 0]))
    pcd4 = make3dset(poi_xyzVecMatTransformed, np.array([0, 1, 0]))
    pcd5 = make3dset(xyzVecMat3, np.array([1, 0, 1]))
    
    xyzVecMat2_full = vid.flattenCombine(xyzMat2)
    xyzVecMat_1in2_full = vid.flattenCombine(xyzMat_1in2)
    
    zLim = 5*1000
    xyzVecMat2_full[xyzVecMat2_full[:,2] > zLim] = 0
    xyzVecMat_1in2_full[xyzVecMat_1in2_full[:,2] > zLim] = 0
    
    pcd6 = make3dset(xyzVecMat2_full, np.array([0, 0.6, 0]))
    pcd7 = make3dset(xyzVecMat_1in2_full, np.array([0.6, 0, 0.6]))  
    
    o3d.visualization.draw_geometries([pcd6, pcd7])
    o3d.visualization.draw_geometries([pcd1, pcd2])
    o3d.visualization.draw_geometries([pcd3, pcd2])
# plotting matches


rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(cornerMatchedIdx_p.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchedIdx_p[0, ii], cornerMatchedIdx_p[1, ii], 8, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchedIdx_c_opt[0, ii], cornerMatchedIdx_c_opt[1, ii], 8, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)

plt.figure('rgb  frame 2 interest points')
plt.title('rgb  frame 2 interest points')
plt.imshow(rgb2Match)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(xyzPoints_p[:,0], xyzPoints_p[:,1], xyzPoints_p[:,2]);
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')










