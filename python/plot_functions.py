import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
def plot2_Open3d(pcd1, pcd2):
    
    #o3d.visualization.draw_geometries([pcd1])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=0)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd1)
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=0)
    opt = vis2.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
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
   
    
#------------------------------------------------------------------------------
def plot3d(xyz):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')   
    s = 0.005     
    ax.scatter3D(xyz[:,0], xyz[:,1], xyz[:,2], s=s, marker=".")
    
    ax.view_init(90, -90)
    
    rg = 4*1e3
    ax.set_xlim(-rg/2, rg/2)
    ax.set_ylim(-rg/2, rg/2)
    ax.set_zlim(0, rg)
    plt.xlabel('x')
    plt.ylabel('y')
    fig.tight_layout()
    plt.show()
    
#------------------------------------------------------------------------------
def plotMask(mat, name):
    
    fig = plt.figure(name)
    plt.title(name)
    plt.spy(mat)