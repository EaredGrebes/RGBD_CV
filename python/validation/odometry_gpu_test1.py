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

plt.close('all')
 
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
frame1 = 160
rgbMat1, xyzMat1, maskMat1 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame1)

height, width = maskMat1.shape
    
#------------------------------------------------------------------------------
# GPU odometry

# initialize some objects
rgbdObj = rgbd.RGBD_odometry_gpu_class(rgbMat1, xyzMat1, maskMat1)

imgTransformObj = imgt.image_transform_class(height, width, rgbdObj.nPoi)

# add a new frame
frame2 = frame1 + 1
rgbMat2, xyzMat2, maskMat2 = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame2)


xyzPoints_p, xyzPoints_c, xyzPoints_pinc = rgbdObj.matchNewFrame(rgbMat2, xyzMat2, maskMat2)


start = time.time()
#------------------------------------------------------------------------------
# prototyping
R_fTobc = rgbdObj.R_fTobc
t_fTobc_inf = rgbdObj.t_fTobc_inf

nOpt = 20
deltaVec = np.zeros((nOpt, 6))
normVec = np.zeros(nOpt)
deltaSum = np.zeros(6)
print(np.linalg.norm(t_fTobc_inf))
for ii in range(nOpt):
    imgTransformObj.computeTransformJacobian(rgbdObj.xyzVecMat_p,         \
                                             rgbdObj.sensorData_c['zMat'], \
                                             R_fTobc,    \
                                             t_fTobc_inf)
        
    jacobianMat = imgTransformObj.jacobianMat.get()
    zErrorPerturbMat = imgTransformObj.zErrorPerturbMat.get()
    zErrorVec = imgTransformObj.zErrorVec.get()
    
    A = jacobianMat.T @ jacobianMat + 0.3 * np.eye(6)
    
    delta = -0.8 * np.linalg.solve(A, jacobianMat.T @ zErrorVec)

    DcmDelta = imgt.eulerAngToDcm(delta[3:])
    
    R_fTobc = DcmDelta @ R_fTobc
    t_fTobc_inf = t_fTobc_inf + delta[:3]
    deltaSum += delta
    
    #print(np.linalg.norm(delta))
    deltaVec[ii, :] = deltaSum
    normVec[ii] = np.sum(np.power(zErrorVec,2))

print(np.linalg.norm(t_fTobc_inf))
print('timer:', time.time() - start)   


plt.figure()
plt.plot(deltaVec)
# update state

plt.figure()
plt.plot(normVec)

plt.figure()
plt.hist(zErrorVec[zErrorVec != 0], bins = 200)

# # construct matrix of perturbed rotation and translations (rotation matrix flattened into 1x9 array)
# deltaAng = 0.0
# deltaPos = 0.0
# transPerturb_cpu, rotPerturb_cpu = imgt.createPerurbedTransforms(deltaAng, deltaPos, rgbdObj.R_fTobc, rgbdObj.t_fTobc_inf)

# transPerturb = cp.array(transPerturb_cpu, dtype = cp.float32)
# rotPerturb = cp.array(rotPerturb_cpu, dtype = cp.float32)

# nTransforms = transPerturb.shape[0]
# zErrorMat = cp.zeros((rgbdObj.xyzVecMat_p.shape[0], nTransforms), dtype = cp.float32)

# imgTransformObj.multiTransformXYZVecMat(zErrorMat,   \
#                                         rgbdObj.xyzVecMat_p, \
#                                         rgbdObj.sensorData_c['zMat'],\
#                                         rotPerturb,\
#                                         transPerturb)       
    
# zErrorMat_cpu = zErrorMat.get() 

#------------------------------------------------------------------------------
# verification
xyzScale = 1
xyzOffset = 0
#nMatchedPoints = rgbdObj.cornerMatchedIdx_c.shape[1]
nMatchedPoints = rgbdObj.nPoi

xyzMatched_prevPoints = cp.zeros((nMatchedPoints, 3), dtype = cp.float32)
pixelIdVec = cp.zeros((nMatchedPoints, 2), dtype = cp.int32)
zErrorVec = cp.zeros((nMatchedPoints), dtype = cp.float32)

# fmgpu.generateXYZVecMat(xyzMatched_prevPoints, \
#                         rgbdObj.sensorData_p['xMat'], \
#                         rgbdObj.sensorData_p['yMat'], \
#                         rgbdObj.sensorData_p['zMat'], \
#                         rgbdObj.sensorData_p['maskMat'], \
#                         rgbdObj.cornerMatchedIdx_p, \
#                         xyzScale,
#                         xyzOffset)

# imgTransformObj.transformXYZVecMat(pixelIdVec, \
#                            zErrorVec,  \
#                            xyzMatched_prevPoints,        \
#                            rgbdObj.sensorData_c['zMat'],\
#                            Dcm,\
#                            translationVec)     
  
#Dcm = cp.array(rgbdObj.R_bpTobc, dtype = cp.float32)
#translation = cp.array(rgbdObj.t_bpTobc_inbc, dtype = cp.float32)

Dcm = cp.array(R_fTobc, dtype = cp.float32)
translation = cp.array(t_fTobc_inf, dtype = cp.float32)

xyzVecMatTransformed = cp.zeros(rgbdObj.xyzVecMat_p.shape, dtype = cp.float32)

imgTransformObj.transformXYZVecMat(xyzVecMatTransformed, \
                                   pixelIdVec, \
                                   zErrorVec,  \
                                   rgbdObj.xyzVecMat_p,         \
                                   rgbdObj.sensorData_c['zMat'],\
                                   Dcm,\
                                   translation)        
 
xyzVecMatTransformed_cpu = xyzVecMatTransformed.get()    
zErrorVec_cpu = zErrorVec.get()
    
cornerMatchedIdx1 = rgbdObj.cornerMatchedIdx_p.get()
cornerMatchedIdx2 = rgbdObj.cornerMatchedIdx_c.get()
cornerMatchedIdx3 = pixelIdVec.T.get()

zMat = rgbdObj.sensorData_c['zMat'].get()
xyzVecMat3 = np.zeros(xyzVecMatTransformed_cpu.shape)

for ii in range(xyzVecMatTransformed_cpu.shape[0]):
    
    xyzVecMat3[ii, 0] = xyzMat2[cornerMatchedIdx3[0,ii], cornerMatchedIdx3[1,ii], 0]
    xyzVecMat3[ii, 1] = xyzMat2[cornerMatchedIdx3[0,ii], cornerMatchedIdx3[1,ii], 1]
    xyzVecMat3[ii, 2] = xyzMat2[cornerMatchedIdx3[0,ii], cornerMatchedIdx3[1,ii], 2]
  
zVec1 = rgbdObj.xyzVecMat_p.get()[:,2]    
zVec2 = xyzVecMat3[:,2]

zErrTest = zVec2 - zVec1


#------------------------------------------------------------------------------
# plotting

plt3d = True
if plt3d:
    nP = xyzPoints_p.shape[0]
    rgbPoints1 = np.zeros((nP, 3))
    rgbPoints2 = np.zeros((nP, 3))
    rgbPoints3 = np.zeros((nP, 3))
    rgbPoints4 = np.zeros(xyzVecMatTransformed_cpu.shape)
    rgbPoints5 = np.zeros(xyzVecMat3.shape)
    
    rgbPoints1[:,0] = 1
    rgbPoints1[:,1] = 0
    rgbPoints1[:,2] = 0
    
    rgbPoints2[:,0] = 0
    rgbPoints2[:,1] = 1
    rgbPoints2[:,2] = 0
    
    rgbPoints3[:,0] = 0
    rgbPoints3[:,1] = 0
    rgbPoints3[:,2] = 1
    
    rgbPoints4[:,0] = 1
    rgbPoints4[:,1] = 0
    rgbPoints4[:,2] = 1
    
    rgbPoints5[:,0] = 0.3
    rgbPoints5[:,1] = 0
    rgbPoints5[:,2] = 1
    
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
    
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(xyzVecMatTransformed_cpu)
    pcd4.colors = o3d.utility.Vector3dVector(rgbPoints4) 
    
    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(xyzVecMat3)
    pcd5.colors = o3d.utility.Vector3dVector(rgbPoints5) 
    
    o3d.visualization.draw_geometries([pcd4, pcd5])

# plotting matches


rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(cornerMatchedIdx1.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchedIdx1[0, ii], cornerMatchedIdx1[1, ii], 8, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchedIdx2[0, ii], cornerMatchedIdx2[1, ii], 8, color.astype(np.ubyte))
    
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










