import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
def plotDual_Open3d(pcd1, pcd2):
    
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
    
#------------------------------------------------------------------------------
def plotRgbMats(redMat, greenMat, blueMat, name):
    
    plt.figure(name)
    plt.title(name)
    plt.imshow(np.concatenate((redMat[:,:,np.newaxis], greenMat[:,:,np.newaxis], blueMat[:,:,np.newaxis]), axis=2))
    plt.show()  
    
    
#------------------------------------------------------------------------------
def plotRgbHistogram(rgb, rgbPdf):   
    
    redHist = np.sum(rgbPdf, axis = (1,2))
    greenHist = np.sum(rgbPdf, axis = (0,2))
    blueHist = np.sum(rgbPdf, axis = (0,1))
    
    xVec = np.linspace(2, 255, len(redHist))
    
    fig, axs = plt.subplots(3, 1)
    
    axs[0].hist(rgb[:,0], density=True, bins=128, color = (1, 0, 0))
    axs[0].step(xVec, redHist/2)
    axs[0].set_title('red histogram')

    axs[1].hist(rgb[:,1], density=True, bins=128, color = (0, 1, 0))
    axs[1].step(xVec, greenHist/2)
    axs[1].set_title('green histogram')

    axs[2].hist(rgb[:,2], density=True, bins=128, color = (0, 0, 1))
    axs[2].step(xVec, blueHist/2)
    axs[2].set_title('blue histogram')

    fig.patch.set_facecolor((0.7, 0.7, 0.7))
    for ax in axs:
        ax.sharex(axs[0])
        ax.set_facecolor((0, 0, 0))

