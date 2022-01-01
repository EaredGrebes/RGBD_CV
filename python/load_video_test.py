import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import video_functions as vid
import time
import h5py
import multiprocessing as mp
import open3d as o3d


print(sys.executable)
print("Number of cpu: ", mp.cpu_count())

loadCal = True
loadVidPar = True

#------------------------------------------------------------------------------
width = 848
height = 480

calName = '../data/calibration.h5'
vidNames = ['../data/videoCaptureTest1.avi', \
            '../data/videoCaptureTest2.avi', \
            '../data/videoCaptureTest3.avi'  ]
    
if loadCal:
    datH5 = h5py.File(calName)
    print(datH5.keys())
    pixelXPosMat = np.array(datH5["pixelXPosMat"][:])
    pixelYPosMat = np.array(datH5["pixelYPosMat"][:])  
    
if loadVidPar:
    start = time.time()
    vidTensorList = vid.loadVideos(vidNames, width, height)
    print('timer:', time.time() - start)
 
#------------------------------------------------------------------------------
    
# video indexes
depth8L = 1
depth8U = 2

nFrms = vidTensorList[depth8L][1]

frame = 10

depth8LMat = vidTensorList[depth8L][0][:,:,frame]
depth8UMat = vidTensorList[depth8U][0][:,:,frame]

xMat = np.zeros((height, width))
yMat = np.zeros((height, width))
zMat = np.zeros((height, width))

for ii in range(height):
    for jj in range(width):
        z = depth8LMat[ii,jj] * 255 + depth8UMat[ii,jj]
        
        xMat[ii,jj] = z * pixelXPosMat[ii,jj]
        yMat[ii,jj] = z * pixelYPosMat[ii,jj]
        zMat[ii,jj] = z
        
X = xMat.flatten()
Y = yMat.flatten()
Z = zMat.flatten()
xyz = np.zeros((np.size(X), 3))
xyz[:, 0] = np.reshape(X, -1)
xyz[:, 1] = np.reshape(Y, -1)
xyz[:, 2] = np.reshape(Z, -1) 


#------------------------------------------------------------------------------
# some plotting
    
plt.close('all')

fig = plt.figure()
ax = plt.axes(projection='3d')   
s = 0.005     
ax.scatter3D(X, -Y, Z, s=s, marker=".")

ax.view_init(90, -90)

rg = 5*1e3
ax.set_xlim(-rg/2, rg/2)
ax.set_ylim(-rg/2, rg/2)
ax.set_zlim(0, rg)
plt.xlabel('x')
plt.ylabel('y')
fig.tight_layout()


# open3d point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.visualization.draw_geometries([pcd])


