import numpy as np
import cupy as cp
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu


#------------------------------------------------------------------------------
  
class RGBD_odometry_gpu_class:

    def __init__(self, height, width, cornerScale, matchScale, nMax, rgbMat1, xyzMat1, maskMat1, rgbMat2, xyzMat2, maskMat2): 
        
        # corner detector object 
        self.cornerObjGpu = fdgpu.corner_detector_class(height, width, cornerScale, nMax)
        
        # match object
        self.matchObjGpu = fmgpu.feature_matching_class(imHeight, imgWidth, nMax, scale)
        
        # previous corner points
        self.cornerIdx_p = cornerObjGpu.findCornerPoints(fdgpu.rgbToGreyMat(rgbMat1), maskMat1_gpu)
        
        # current corner points
        self.cornerIdx_c = cornerObjGpu.findCornerPoints(fdgpu.rgbToGreyMat(rgbMat2), maskMat2_gpu)
        
        
    def genMatchMatrix(self, idx1, idx2):
        
        
        
        
        
    def addNewFrame(self, rgbMat, xyzMat, maskMat):
        
        tmp = 3
        



#------------------------------------------------------------------------------

