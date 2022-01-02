import numpy as np
import cv2
import video_functions as vid
import multiprocessing as mp
import open3d as o3d


#------------------------------------------------------------------------------
# loads multiple videos in parallel
def loadVideos(videos, width, height):
    
    inputList = []
    for vd in videos:
        inputList.append([vd['filename'], vd['channel'], width, height])    

    pool = mp.Pool(processes = len(videos))
    vidTensorList = pool.map(vid.genVideoTensor, inputList)
    pool.close()  
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
    #while(cap.isOpened()):
    for ii in range(40):
        
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
def genXYZ(depth8LMat, depth8UMat, pixelXPosMat, pixelYPosMat, width, height):
    
    xMat = np.zeros((height, width))
    yMat = np.zeros((height, width))
    zMat = np.zeros((height, width))

    for ii in range(height):
        for jj in range(width):
            z = depth8LMat[ii,jj] * 255 + depth8UMat[ii,jj]
            
            xMat[ii,jj] = z * pixelXPosMat[ii,jj]
            yMat[ii,jj] = z * pixelYPosMat[ii,jj]
            zMat[ii,jj] = z
            
    return xMat, yMat, zMat


#------------------------------------------------------------------------------
def genPointCloud(xMat, yMat, zMat, redMat, greenMat, blueMat):
    
    def flattenCombine(Mat1, Mat2, Mat3): 
        X = Mat1.flatten()
        Y = Mat2.flatten()
        Z = Mat3.flatten()
        xyz = np.zeros((np.size(X), 3))
        xyz[:, 0] = np.reshape(X, -1)
        xyz[:, 1] = np.reshape(Y, -1)
        xyz[:, 2] = np.reshape(Z, -1) 
        
        return xyz
    
    xyz = flattenCombine(xMat, yMat, zMat)
    rgb = flattenCombine(redMat, greenMat, blueMat)
    rgb = rgb / 255
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd
    
#------------------------------------------------------------------------------
def getFrame(vidTensorList, vdid, pixelXPosMat, pixelYPosMat, width, height, frame):
    print('getting frame: ', frame)
    
    redMat   = vidTensorList[vdid['red']][:,:,frame]
    greenMat = vidTensorList[vdid['green']][:,:,frame]
    blueMat  = vidTensorList[vdid['blue']][:,:,frame]
    depth8LMat = vidTensorList[vdid['depth8L']][:,:,frame]
    depth8UMat = vidTensorList[vdid['depth8U']][:,:,frame]

    xMat, yMat, zMat = vid.genXYZ(depth8LMat, \
                                  depth8UMat, \
                                  pixelXPosMat, \
                                  pixelYPosMat, \
                                  width, \
                                  height)
        
    pcd = vid.genPointCloud(xMat, yMat, zMat, redMat, greenMat, blueMat)  
    
    return pcd, xMat, yMat, zMat, redMat, greenMat, blueMat
