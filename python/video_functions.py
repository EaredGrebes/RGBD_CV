import numpy as np
import cv2
import video_functions as vid
import multiprocessing as mp

#------------------------------------------------------------------------------
# loads multiple videos in parallel
def loadVideos(vidNames, width, height):
    
    inputList = []
    nVids = len(vidNames)
    for ii in range(nVids):
        inputList.append([vidNames[ii], width, height])    

    pool = mp.Pool(processes = nVids)
    vidTensorList = pool.map(vid.genVideoTensor, inputList)
    pool.close()  
    pool.join() 
    
    return vidTensorList
    

#------------------------------------------------------------------------------
# generates a tensor from a video
def genVideoTensor(inArgs):
    
    vidName = inArgs[0]
    width   = inArgs[1]
    height  = inArgs[2]
    
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
            frameMat[:,:] = np.asarray(frame[:,:,0])
            frames.append(frameMat)
            frameCount += 1
        else:
            break
    cap.release()

    # loop through the list, and form a tensor containing each frame
    frameTensor = np.ndarray((height, width, frameCount), dtype = np.ubyte)
    for ii in range(frameCount):
        frameTensor[:,:,ii] = frames[ii]
     
    return frameTensor, frameCount

