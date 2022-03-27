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
frame1 = 240
rgbMat1, xyzMat1, maskMat1 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)

height, width = maskMat1.shape
    
#------------------------------------------------------------------------------
# GPU implementation testing

rgbdObj = rgbd.RGBD_odometry_gpu_class(rgbMat1, xyzMat1, maskMat1)

frame2 = frame1 + 20
rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)

xyzPoints_p, xyzPoints_c, xyzPoints_pinc = rgbdObj.matchNewFrame(rgbMat2, xyzMat2, maskMat2)

# verification
xyzScale = 1
xyzOffset = 0
nMatchedPoints = rgbdObj.cornerMatchedIdx_c.shape[1]

xyzVecMat_prevPoints = cp.zeros((nMatchedPoints, 3), dtype = cp.float32)
pixelIdVec = cp.zeros((nMatchedPoints, 2), dtype = cp.int32)

fmgpu.generateXYZVecMat(xyzVecMat_prevPoints, \
                        rgbdObj.sensorData_p['xMat'], \
                        rgbdObj.sensorData_p['yMat'], \
                        rgbdObj.sensorData_p['zMat'], \
                        rgbdObj.sensorData_p['maskMat'], \
                        rgbdObj.cornerMatchedIdx_p, \
                        xyzScale,
                        xyzOffset)
            
imgtObj = imgt.image_transform_class()

R = cp.array(rgbdObj.R_fTobc, dtype = cp.float32)
t = cp.array(rgbdObj.t_fTobc_inf, dtype = cp.float32)

#R = cp.eye(3, dtype = cp.float32)
#t = cp.zeros((3), dtype = cp.float32)

imgtObj.transformXYZVecMat(pixelIdVec, xyzVecMat_prevPoints, R, t)        
  
    
cornerMatchedIdx1 = rgbdObj.cornerMatchedIdx_p.get()
cornerMatchedIdx2 = rgbdObj.cornerMatchedIdx_c.get()
cornerMatchedIdx3 = pixelIdVec.T.get()


#------------------------------------------------------------------------------
# plotting
plt.close('all')

nP = xyzPoints_p.shape[0]
rgbPoints1 = np.zeros((nP, 3))
rgbPoints2 = np.zeros((nP, 3))
rgbPoints3 = np.zeros((nP, 3))

rgbPoints1[:,0] = 1
rgbPoints1[:,1] = 0
rgbPoints1[:,2] = 0

rgbPoints2[:,0] = 0
rgbPoints2[:,1] = 1
rgbPoints2[:,2] = 0

rgbPoints3[:,0] = 0
rgbPoints3[:,1] = 0
rgbPoints3[:,2] = 1

# open3d
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(xyzPoints_p)
pcd1.colors = o3d.utility.Vector3dVector(rgbPoints1)  # open3d expects color between [0, 1]

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(xyzPoints_c)
pcd2.colors = o3d.utility.Vector3dVector(rgbPoints2) 

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(xyzPoints_pinc)
pcd3.colors = o3d.utility.Vector3dVector(rgbPoints3) 

o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

# plotting matches


rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(cornerMatchedIdx1.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchedIdx1[0, ii], cornerMatchedIdx1[1, ii], 8, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchedIdx3[0, ii], cornerMatchedIdx3[1, ii], 8, color.astype(np.ubyte))
    
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










