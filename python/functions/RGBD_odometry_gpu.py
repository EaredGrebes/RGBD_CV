import numpy as np
import cupy as cp
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu
import image_transform_functions_gpu as tmgpu
import image_transform_functions_gpu as imgt
import copy
import time


#------------------------------------------------------------------------------
  
class RGBD_odometry_gpu_class:

    def __init__(self, rgbMat_init, xyzMat_init, maskMat_init): 
        
        # ~~ parameters  ~~
        self.nFeatures = 128
        self.cornerScale = 8
        self.matchScale = 13
        self.matchOffset = np.int32(self.matchScale/2)
        l = 7
        self.xyzScale = 2*l+1  # side length of pixel box that goes around a feature point to generate multiple points of interest
        self.xyzOffset = l
        self.transformPercent = 0.6  # top percentage of 3d points used in ICP transform
        
        self.zMin_mm = 200
        
        height, width = maskMat_init.shape
        self.nPoi = self.nFeatures * self.xyzScale * self.xyzScale # number of 3d points of interest
        
        # ~~ objects  ~~
        # corner detector object 
        self.cornerObjGpu = fdgpu.corner_detector_class(height, width, self.cornerScale, self.nFeatures)

        # matching object
        self.matchObjGpu = fmgpu.feature_matching_class(height, width, self.nFeatures, self.matchScale)
        
        # transform object
        self.transformObjGpu = tmgpu.image_transform_class(height, width, self.nPoi)
        
        # initial sensor data
        self.sensorData_p = generateGpuSensorDataDict(xyzMat_init, rgbMat_init, maskMat_init)
        self.sensorData_c = generateGpuSensorDataDict(xyzMat_init, rgbMat_init, maskMat_init)
        
        # ~~ working variables  ~~
        # a dictionary of matrcies that only store points of interest
        self.poiMats_p = generatePoiMats(self.nFeatures, self.matchScale, height, width)
        self.poiMats_c = generatePoiMats(self.nFeatures, self.matchScale, height, width)
        
        self.cornerMatchedIdx_p = None # variable size
        self.cornerMatchedIdx_c = None

        self.pixelIdTransformedVec = cp.zeros((self.nPoi, 3), dtype = cp.float32)
        
        self.xyzVecMat_p = cp.zeros((self.nPoi, 3), dtype = cp.float32)
        self.greyVec_p = cp.zeros((self.nPoi), dtype = cp.float32)
        self.xyzVecMat_c = cp.zeros((self.nPoi, 3), dtype = cp.float32)
        self.greyVec_c = cp.zeros((self.nPoi), dtype = cp.float32)
        self.xyzVecMat_c_matched = cp.zeros((self.nPoi, 3), dtype = cp.float32)
        
        self.R_bpTobc = np.eye(3)
        self.t_bpTobc_inbc = np.zeros((3))
        self.R_fTobc = np.eye(3)
        self.t_fTobc_inf = np.zeros((3))
        
        # process the initial data
        self.processNewSensorData(self.sensorData_c, self.poiMats_c)
        
        
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
                                self.greyVec_p ,  \
                                self.sensorData_p['xMat'], \
                                self.sensorData_p['yMat'], \
                                self.sensorData_p['zMat'], \
                                self.sensorData_p['greyMat'], \
                                self.sensorData_p['maskMat'], \
                                self.cornerMatchedIdx_p, \
                                self.xyzScale,
                                self.xyzOffset)  
        
        fmgpu.generateXYZVecMat(self.xyzVecMat_c, \
                                self.greyVec_c,   \
                                self.sensorData_c['xMat'], \
                                self.sensorData_c['yMat'], \
                                self.sensorData_c['zMat'], \
                                self.sensorData_p['greyMat'], \
                                self.sensorData_c['maskMat'], \
                                self.cornerMatchedIdx_c, \
                                self.xyzScale,
                                self.xyzOffset)    

        # use matched XYZ matricies to compute rigid transform between frames
        Nmatches = self.cornerMatchedIdx_p.shape[1]
        # print('Nmatches: {}'.format(Nmatches))
        
        xyzPoints_p = self.xyzVecMat_p[:Nmatches*self.xyzScale*self.xyzScale,:].get()
        xyzPoints_c = self.xyzVecMat_c[:Nmatches*self.xyzScale*self.xyzScale,:].get()

        pointMask = (abs(xyzPoints_p[:,2]) > self.zMin_mm) & (abs(xyzPoints_c[:,2]) > self.zMin_mm)

        xyzPoints_p = xyzPoints_p[pointMask,:]
        xyzPoints_c = xyzPoints_c[pointMask,:]
        Npoints = xyzPoints_c.shape[0]
        
        # solve transform using all valid points
        _, _, distErr, _ = solveTransform(xyzPoints_p, xyzPoints_c, Npoints)

        nSmallestPoints = int(Npoints * self.transformPercent)
        idxSmallestDist = np.argsort(distErr)[:nSmallestPoints]
        
        # solve transform using the points with the smallest error (throwing out outliers), cheap RANSAC
        R_bpTobc, t_bpTobc_inbc, distErr2, xyzPoints_pinc = solveTransform(xyzPoints_p[idxSmallestDist,:], xyzPoints_c[idxSmallestDist,:], nSmallestPoints)
        
        # adjust the transform by solving a global optimization problem
        R_bpTobc, t_bpTobc_inbc, optDeltaVec, optNormVec, optCostVec = self.solveOptimalTransform(R_bpTobc, t_bpTobc_inbc)
        
        self.R_bpTobc = R_bpTobc
        self.t_bpTobc_inbc = t_bpTobc_inbc
        
        # update camera DCM and translation (cpu)
        self.R_fTobc = self.R_bpTobc @ self.R_fTobc
        self.t_fTobc_inf = self.t_fTobc_inf + self.R_fTobc.T @ self.t_bpTobc_inbc

        return optDeltaVec, optNormVec, optCostVec, xyzPoints_p[idxSmallestDist,:], xyzPoints_c[idxSmallestDist,:], xyzPoints_pinc
        


    def solveOptimalTransform(self, R_fTobc, t_fTobc_inf):
        
        Dcm = np.copy(R_fTobc)
        translation = np.copy(t_fTobc_inf)

        nOpt = 60
        deltaVec = np.zeros((nOpt, 6))
        normVec = np.zeros(nOpt)
        deltaSum = np.zeros(6)
        
        for ii in range(nOpt):
            jacobianMat_gpu, costVec_gpu = self.transformObjGpu.computeTransformJacobian(self.xyzVecMat_p,         \
                                                                                         self.greyVec_p,            \
                                                                                         self.sensorData_c['zMat'], \
                                                                                         self.sensorData_c['greyMat'], \
                                                                                         Dcm,    \
                                                                                         translation)
            
            
            jacobianMat = jacobianMat_gpu.get()
            costVec = costVec_gpu.get()
            
            A = jacobianMat.T @ jacobianMat 
            
            delta = -0.95 * np.linalg.solve(A, jacobianMat.T @ costVec)

            DcmDelta = imgt.eulerAngToDcm(delta[3:])
            Dcm = DcmDelta @ Dcm
            
            translation = translation + delta[:3]
            deltaSum += delta
            #DcmDelta = imgt.eulerAngToDcm(deltaSum[3:])
            #R_fTobc = DcmDelta @ rgbdObj.R_fTobc
            
            #print(np.linalg.norm(delta))
            deltaVec[ii, :] = deltaSum
            normVec[ii] = np.sum(np.power(costVec,2))
            
        return Dcm, translation, deltaVec, normVec, costVec
            
        
            
            
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
    