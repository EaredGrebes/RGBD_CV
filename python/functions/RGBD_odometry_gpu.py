import numpy as np
import cupy as cp
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu
import image_transform_functions_gpu as tmgpu
import copy
import time



#------------------------------------------------------------------------------
  
class RGBD_odometry_gpu_class:

    def __init__(self, rgbMat_init, xyzMat_init, maskMat_init): 
        
        # ~~ parameters  ~~
        self.nFeatures = 128
        self.cornerScale = 8
        self.matchScale = 15
        self.matchOffset = np.int32(self.matchScale/2)
        l = 5
        self.xyzScale = 2*l+1  # side length of box that goes around point of interest x,y,z
        self.xyzOffset = l
        self.transformPercent = 0.3
        
        height, width = maskMat_init.shape
        
        # ~~ objects  ~~
        # corner detector object 
        self.cornerObjGpu = fdgpu.corner_detector_class(height, width, self.cornerScale, self.nFeatures)

        # matching object
        self.matchObjGpu = fmgpu.feature_matching_class(height, width, self.nFeatures, self.matchScale)
        
        # transform object
        self.transformObjGpu = tmgpu.image_transform_class()
        
        # initial sensor data
        self.sensorData_p = generateGpuSensorDataDict(xyzMat_init, rgbMat_init, maskMat_init)
        self.sensorData_c = generateGpuSensorDataDict(xyzMat_init, rgbMat_init, maskMat_init)
        
        # ~~ working variables  ~~
        # only pixel data around point of interest
        self.poiMats_p = generatePoiMats(self.nFeatures, self.matchScale, height, width)
        self.poiMats_c = generatePoiMats(self.nFeatures, self.matchScale, height, width)
        
        self.processNewSensorData(self.sensorData_c, self.poiMats_c)
        
        # initialize later after matchNewFrame is called
        self.cornerMatchedIdx_p = None
        self.cornerMatchedIdx_c = None
        self.pixelIdTransformedVec = cp.zeros((self.nFeatures * self.xyzScale * self.xyzScale, 3), dtype = cp.float32)
        
        self.xyzVecMat_p = cp.zeros((self.nFeatures * self.xyzScale * self.xyzScale, 3), dtype = cp.float32)
        self.xyzVecMat_c = cp.zeros((self.nFeatures * self.xyzScale * self.xyzScale, 3), dtype = cp.float32)
        self.xyzVecMat_c_matched = cp.zeros((self.nFeatures * self.xyzScale * self.xyzScale, 3), dtype = cp.float32)
        
        self.R_fTobc = np.eye(3)
        self.t_fTobc_inf = np.zeros((3))
        
    #--------------------------------------------------------------------------
    def processNewSensorData(self, sensorData, poiMats):
        
        poiMats['xMat'] = sensorData['xMat']
        poiMats['yMat'] = sensorData['yMat']
        poiMats['zMat'] = sensorData['zMat']
        poiMats['maskMat'] = sensorData['maskMat']
        
        self.cornerObjGpu.findCornerPoints(poiMats['cornerPointIdx'], sensorData['greyMat'], sensorData['maskMat'])
        
        fmgpu.generatePointsOfInterestMat(poiMats['poi_rMat'], \
                                          poiMats['poi_gMat'], \
                                          poiMats['poi_bMat'], \
                                          poiMats['poi_maskMat'], \
                                          sensorData['rMat'], \
                                          sensorData['gMat'], \
                                          sensorData['bMat'], \
                                          sensorData['maskMat'], \
                                          poiMats['cornerPointIdx'],
                                          self.matchScale,
                                          self.matchOffset)

            
    #--------------------------------------------------------------------------
    def matchNewFrame(self, rgbMat, xyzMat, maskMat):
        
        # save current poi data to previous
        self.sensorData_p = copy.deepcopy(self.sensorData_c)
        self.poiMats_p = copy.deepcopy(self.poiMats_c)
        
        # convert new sensor data to current poi data
        setGpuSensorDataDict(self.sensorData_c, xyzMat, rgbMat, maskMat)
        self.processNewSensorData(self.sensorData_c, self.poiMats_c)
        
        # compute poi feature matches between current and previous
        self.cornerMatchedIdx_p, \
        self.cornerMatchedIdx_c = self.matchObjGpu.computeMatches( self.poiMats_p['poi_rMat'], \
                                                                   self.poiMats_p['poi_gMat'], \
                                                                   self.poiMats_p['poi_bMat'], \
                                                                   self.poiMats_p['poi_maskMat'], \
                                                                   self.poiMats_p['cornerPointIdx'], \
                                                                   self.poiMats_c['poi_rMat'], \
                                                                   self.poiMats_c['poi_gMat'], \
                                                                   self.poiMats_c['poi_bMat'], \
                                                                   self.poiMats_c['poi_maskMat'], \
                                                                   self.poiMats_c['cornerPointIdx'])
            
        # construct the matched XYZ matricies from the current and previous data
        fmgpu.generateXYZVecMat(self.xyzVecMat_p, \
                                self.sensorData_p['xMat'], \
                                self.sensorData_p['yMat'], \
                                self.sensorData_p['zMat'], \
                                self.sensorData_p['maskMat'], \
                                self.cornerMatchedIdx_p, \
                                self.xyzScale,
                                self.xyzOffset)  
        
        fmgpu.generateXYZVecMat(self.xyzVecMat_c, \
                                self.sensorData_c['xMat'], \
                                self.sensorData_c['yMat'], \
                                self.sensorData_c['zMat'], \
                                self.sensorData_c['maskMat'], \
                                self.cornerMatchedIdx_c, \
                                self.xyzScale,
                                self.xyzOffset)    

        # use matches XYZ matricies to compute rigid transform between frames
        Nmatches = self.cornerMatchedIdx_p.shape[1]
        # print('Nmatches: {}'.format(Nmatches))
        
        
        xyzPoints_p = self.xyzVecMat_p[:Nmatches*self.xyzScale*self.xyzScale,:].get()
        xyzPoints_c = self.xyzVecMat_c[:Nmatches*self.xyzScale*self.xyzScale,:].get()
        
        pointMask = (abs(xyzPoints_p[:,2]) > 0) & (abs(xyzPoints_c[:,2]) > 0)

        xyzPoints_p = xyzPoints_p[pointMask,:]
        xyzPoints_c = xyzPoints_c[pointMask,:]
        Npoints = xyzPoints_c.shape[0]
        
        _, _, distErr, _ = solveTransform(xyzPoints_p, xyzPoints_c, Npoints)

        nSmallestPoints = int(Npoints * self.transformPercent)
        idxSmallestDist = np.argsort(distErr)[:nSmallestPoints]
        R_bpTobc, t_bpTobc_inbc, distErr2, xyzPoints_pinc = solveTransform(xyzPoints_p[idxSmallestDist,:], xyzPoints_c[idxSmallestDist,:], nSmallestPoints)
        
        
        
        # CPU: update camera DCM and translation 
        self.R_fTobc = R_bpTobc @ self.R_fTobc
        self.t_fTobc_inf = self.t_fTobc_inf + self.R_fTobc.T @ t_bpTobc_inbc
        
        return xyzPoints_p[idxSmallestDist,:], xyzPoints_c[idxSmallestDist,:], xyzPoints_pinc
        

#------------------------------------------------------------------------------
# helper functions

def generateGpuSensorDataDict(xyzMat, rgbMat, maskMat):

    gpuDataDict = {
        'xMat': cp.array(xyzMat[:,:,0], dtype = cp.float32),
        'yMat': cp.array(xyzMat[:,:,1], dtype = cp.float32),
        'zMat': cp.array(xyzMat[:,:,2], dtype = cp.float32),   
        'rMat': cp.array(rgbMat[:,:,0], dtype = cp.float32),
        'gMat': cp.array(rgbMat[:,:,1], dtype = cp.float32),
        'bMat': cp.array(rgbMat[:,:,2], dtype = cp.float32),
        'greyMat': None,
        'maskMat': cp.array(maskMat, dtype = bool)    
    }
    gpuDataDict['greyMat'] = fdgpu.rgbToGreyMat(gpuDataDict['rMat'], gpuDataDict['gMat'], gpuDataDict['bMat'])
    return gpuDataDict


def setGpuSensorDataDict(gpuDataDict, xyzMat, rgbMat, maskMat):

    gpuDataDict['xMat'] = cp.array(xyzMat[:,:,0], dtype = cp.float32)
    gpuDataDict['yMat'] = cp.array(xyzMat[:,:,1], dtype = cp.float32)
    gpuDataDict['zMat'] = cp.array(xyzMat[:,:,2], dtype = cp.float32)
    
    gpuDataDict['rMat'] = cp.array(rgbMat[:,:,0], dtype = cp.float32)
    gpuDataDict['gMat'] = cp.array(rgbMat[:,:,1], dtype = cp.float32)
    gpuDataDict['bMat'] = cp.array(rgbMat[:,:,2], dtype = cp.float32)
    gpuDataDict['maskMat'] =  cp.array(maskMat, dtype = bool)  
    gpuDataDict['greyMat'] = fdgpu.rgbToGreyMat(gpuDataDict['rMat'], gpuDataDict['gMat'], gpuDataDict['bMat'])


def generatePoiMats(nFeatures, matchScale, height, width):

    poiMats = {
        'poi_rMat': cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32),
        'poi_gMat': cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32),
        'poi_bMat': cp.zeros((nFeatures*matchScale, matchScale), dtype = cp.float32),
        'poi_maskMat': cp.zeros((nFeatures*matchScale, matchScale), dtype = bool),
        'cornerPointIdx': cp.zeros((2, nFeatures), dtype = cp.int32)
    }
    return poiMats
   

def computeVectorDistances(xyzPoints1, xyzPoints2):
    
    xyzError = xyzPoints1 - xyzPoints2
    xyzDist = np.sqrt(np.sum(xyzError*xyzError, axis = 1))
    return xyzDist
    

def solveTransform(xyzPoints1, xyzPoints2, nPoints):
    
    Npoints = xyzPoints1.shape[0]
    mean1 = xyzPoints1.mean(axis = 0)
    mean2 = xyzPoints2.mean(axis = 0)

    xyzPoints1_c = xyzPoints1 - mean1[None,:]
    xyzPoints2_c = xyzPoints2 - mean2[None,:]

    # cross covariance matrix
    C = (1/nPoints) * xyzPoints1_c.T @ xyzPoints2_c
    U, S, VT = np.linalg.svd(C, full_matrices=True)

    R = VT.T @ U.T
    t = mean2 - R @ mean1

    # apply transform
    xyzPoints1_2 = t[None,:].T + R @ xyzPoints1.T
    xyzPoints1_2 = xyzPoints1_2.T
    
    distanceErrors = computeVectorDistances(xyzPoints1_2, xyzPoints2)
    
    return R, t, distanceErrors, xyzPoints1_2
    