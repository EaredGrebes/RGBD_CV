import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import video_functions as vid
import time
import h5py
import multiprocessing as mp
import open3d as o3d
import functools as ft


print(sys.executable)
print("Number of cpu: ", mp.cpu_count())

plotOpen3d = True

#------------------------------------------------------------------------------
# load data
width = 848
height = 480

# calibration data
calName = '../data/calibration.h5'

datH5 = h5py.File(calName)
print(datH5.keys())
pixelXPosMat = np.array(datH5["pixelXPosMat"][:])
pixelYPosMat = np.array(datH5["pixelYPosMat"][:])  

# video streams
vdid = {'blue': 0, 'green': 1, 'red': 2, 'depth8L': 3, 'depth8U': 4}

videos = [{'filename': '../data/videoCaptureTest1.avi', 'channel': 0}, \
          {'filename': '../data/videoCaptureTest1.avi', 'channel': 1}, \
          {'filename': '../data/videoCaptureTest1.avi', 'channel': 2}, \
          {'filename': '../data/videoCaptureTest2.avi', 'channel': 0}, \
          {'filename': '../data/videoCaptureTest3.avi', 'channel': 0}]    

start = time.time()
vidTensorList = vid.loadVideos(videos, width, height)
print('timer:', time.time() - start)
 
#------------------------------------------------------------------------------
# do some processing
(h, w, nFrms) = vidTensorList[vdid['depth8L']].shape
print('number of frames: ', nFrms)

funGetFrame = ft.partial(vid.getFrame, vidTensorList, vdid, pixelXPosMat, pixelYPosMat, width, height)

# get point cloud data for the first and last frame
pcd1, xMat, yMat, zMat, redMat, greenMat, blueMat = funGetFrame(0)
pcd2, xMat, yMat, zMat, redMat, greenMat, blueMat = funGetFrame(nFrms - 1)


#------------------------------------------------------------------------------
# some plotting
plt.close('all')

if plotOpen3d:
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=0)
    vis.add_geometry(pcd1)
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=0)
    vis2.add_geometry(pcd2)
    
    while True:
        #vis.update_geometry()
        if not vis.poll_events():
            break
        vis.update_renderer()
    
        #vis2.update_geometry()
        if not vis2.poll_events():
            break
        vis2.update_renderer()
    
    vis.destroy_window()
    vis2.destroy_window() 
    
   #o3d.visualization.draw_geometries([pcd1])
   
else:
    
    fig = plt.figure()
    plt.spy(vidTensorList[red][:,:,frame])

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
    plt.show()




