import numpy as np
import cv2
import multiprocessing as mp
import open3d as o3d
from os.path import exists
import time
import h5py

#------------------------------------------------------------------------------
# loads multiple videos in parallel
def loadDataSet(videoDat, vdid, calName, numpyName):
    
    # load numpy file, it's faster
    if exists(numpyName):
        print('data.npz exists')
        
        filez = np.load(numpyName, allow_pickle=True)
        
        pixelXPosMat = filez['pixelXPosMat']
        pixelYPosMat = filez['pixelYPosMat']
        redTens = filez['redTens']
        greenTens = filez['greenTens']
        blueTens = filez['blueTens']
        xTens = filez['xTens']
        yTens = filez['yTens']
        zTens = filez['zTens']
        maskTens = filez['maskTens']
       
    # no numpy file, load data set from source videos ans .h5 file, then save as numpy
    else:
        print('data.npz does not exist')
        
        # load cal data
        datH5 = h5py.File(calName)
        print(datH5.keys())
        pixelXPosMat = np.array(datH5["pixelXPosMat"][:])
        pixelYPosMat = np.array(datH5["pixelYPosMat"][:]) 
        height, width = pixelXPosMat.shape
        
        # load video list
        vidTensorList = loadVideos(videoDat, width, height)
        
        # seperate out the different channels / videos
        redTens   = vidTensorList[vdid['red']]
        greenTens = vidTensorList[vdid['green']]
        blueTens  = vidTensorList[vdid['blue']]
        depth8LTens = vidTensorList[vdid['depth8L']]
        depth8UTens = vidTensorList[vdid['depth8U']]
        
        # construct the x,y,z position tensors from the depth 8L and 8U, and pixel calibration data
        height, width, nFrames = depth8UTens.shape
        xTens = np.zeros((height, width, nFrames),  dtype = np.single)
        yTens = np.zeros((height, width, nFrames),  dtype = np.single)
        zTens = np.zeros((height, width, nFrames),  dtype = np.single)
        maskTens = np.zeros((height, width, nFrames),  dtype = bool)
        
        zMin_mm = 200
        zMax_mm = 30 * 1000
        for frame in range(nFrames):
            
            # x,y,z tensors
            zMat = (depth8LTens[:,:,frame].astype(np.single) * 255 + depth8UTens[:,:,frame].astype(np.single))
            zTens[:,:,frame] = zMat
            xTens[:,:,frame] = zMat * pixelXPosMat
            yTens[:,:,frame] = zMat * pixelYPosMat
            
            # mask tensor
            maskMat = np.full((height, width), True, dtype=bool)
            tmp3 = np.logical_or((redTens[:,:,frame] + greenTens[:,:,frame] + blueTens[:,:,frame]) < 15, zMat < zMin_mm, zMat > zMax_mm )
            maskMat[tmp3] = False
            maskTens[:,:,frame] = maskMat
        
        # save data as numpy file for quicker loading
        np.savez(numpyName, pixelXPosMat = pixelXPosMat, 
                            pixelYPosMat = pixelYPosMat,
                            redTens = redTens,
                            greenTens = greenTens,
                            blueTens = blueTens,
                            xTens = xTens,
                            yTens = yTens,
                            zTens = zTens, 
                            maskTens = maskTens) 
        
    return redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens

#------------------------------------------------------------------------------
# loads multiple videos in parallel
def loadVideos(videos, width, height):
    
    inputList = []
    for vd in videos:
        inputList.append([vd['filename'], vd['channel'], width, height])    

    pool = mp.Pool(processes = len(videos))
    vidTensorList = pool.map(genVideoTensor, inputList)
    
    pool.terminate()     
    pool.join() 
    
    return vidTensorList
    
#------------------------------------------------------------------------------
# generates a tensor from a video
def genVideoTensor(inArgs):
    
    vidName = inArgs[0]
    channel = inArgs[1]
    width   = inArgs[2]
    height  = inArgs[3]
    
    print(vidName)
    print(channel)
    
    frameMat = np.ndarray((height, width), dtype = np.ubyte)
    frames = []
    cap = cv2.VideoCapture(vidName)
    frameCount = 0
    
    # loop over each frame, adding it to a list
    print('loading video, might take a while')
    while(cap.isOpened()):
    #for ii in range(40):
        
        ret, frame = cap.read()
        if ret:
            #frameMat[:,:] = np.asarray(frame[:,:,channel])
            frames.append(np.asarray(frame[:,:,channel]))
            frameCount += 1
            #cv2.imshow('Frame',frame)

        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # loop through the list, and form a tensor containing each frame
    frameTensor = np.ndarray((height, width, frameCount), dtype = np.ubyte)
    for ii in range(frameCount):
        frameTensor[:,:,ii] = frames[ii]
     
    return frameTensor


#------------------------------------------------------------------------------
def flattenCombine(Mat3): 
    
    X = Mat3[:,:,0].flatten()
    Y = Mat3[:,:,1].flatten()
    Z = Mat3[:,:,2].flatten()
    xyz = np.zeros((np.size(X), 3))
    xyz[:, 0] = np.reshape(X, -1)
    xyz[:, 1] = np.reshape(Y, -1)
    xyz[:, 2] = np.reshape(Z, -1) 
    
    return xyz

#------------------------------------------------------------------------------
def getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame):
    
    rgbMat = np.stack((redTens[:,:,frame], greenTens[:,:,frame], blueTens[:,:,frame]), axis=2)
    xyzMat = np.stack((xTens[:,:,frame], yTens[:,:,frame], zTens[:,:,frame]), axis=2)
    maskMat = maskTens[:,:,frame]
    
    return rgbMat, xyzMat, maskMat


#------------------------------------------------------------------------------
def genPointCloud(xyzMat, rgbMat, maskMat):
    
    maskFlat = maskMat.flatten()
    xyz = flattenCombine(xyzMat)
    rgb = flattenCombine(rgbMat)
    
    xyz = xyz[maskFlat,:]
    rgb = rgb[maskFlat,:]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return xyz, rgb


#------------------------------------------------------------------------------
def genOpen3dPointCloud(xyzMat, rgbMat, maskMat):
    
    xyz, rgb = genPointCloud(xyzMat, rgbMat, maskMat)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255)  # open3d expects color between [0, 1]
    
    return pcd, xyz, rgb




