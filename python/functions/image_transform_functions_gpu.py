import numpy as np
import cupy as cp
                   
#------------------------------------------------------------------------------
cp_transform_xyzPoints = cp.RawKernel(r'''
extern "C" __global__
void transform_xyzPoints(float* xyzVecMat_transformed, // outputs
                         int* pixelIdVec,     
                         float* zErrorVec,
                         float* xyzVecMat,            // inputs
                         float* zMat,
                         float* rotationMat,
                         float* translationVec,
                         int imgHeight,
                         int imgWidth,
                         float ppx,
                         float ppy,
                         float fx,
                         float fy,
                         int nPoints) {
                    
 
    int iRow = blockIdx.x * blockDim.x  + threadIdx.x;
  
    if (iRow < nPoints) {
                
        // point
        float x1 = xyzVecMat[iRow*3 + 0];
        float y1 = xyzVecMat[iRow*3 + 1];
        float z1 = xyzVecMat[iRow*3 + 2]; 
            
        // transform point
        float x2 = rotationMat[0] * x1 + rotationMat[1] * y1 + rotationMat[2] * z1 + translationVec[0];
        float y2 = rotationMat[3] * x1 + rotationMat[4] * y1 + rotationMat[5] * z1 + translationVec[1];
        float z2 = rotationMat[6] * x1 + rotationMat[7] * y1 + rotationMat[8] * z1 + translationVec[2];
  
        xyzVecMat_transformed[iRow*3 + 0] = x2;
        xyzVecMat_transformed[iRow*3 + 1] = y2;
        xyzVecMat_transformed[iRow*3 + 2] = z2;
        
        // compute point new pixel x,y coordinates in transformed frame
        float xPixel = x2 / z2;
        float yPixel = y2 / z2;
        
        int pixelColId = __float2int_rn(xPixel * fx + ppx);
        int pixelRowId = __float2int_rn(yPixel * fy + ppy);
        
        if( (pixelRowId >= 0) && (pixelRowId < imgHeight) && (pixelColId >= 0) && (pixelColId < imgWidth) ) {
                
            pixelIdVec[iRow*2 + 0] = pixelRowId;
            pixelIdVec[iRow*2 + 1] = pixelColId;
            
            zErrorVec[iRow] = zMat[pixelRowId * imgWidth + pixelColId] - z2;
        }
    }
}
''', 'transform_xyzPoints')


cp_multiTransform_xyzPoints = cp.RawKernel(r'''
extern "C" __global__
void multipleTransform_xyzPoints(float* zErrorMat,    // outputs
                                 float* xyzVecMat,    // inputs
                                 float* zMat,
                                 float* rotationMat,
                                 float* translationVec,
                                 int imgHeight,
                                 int imgWidth,
                                 float ppx,
                                 float ppy,
                                 float fx,
                                 float fy,
                                 int nPoints,
                                 int nTransforms) {
                    
    int iRow = blockIdx.x * blockDim.x  + threadIdx.x;
    int iCol = blockIdx.y * blockDim.y  + threadIdx.y;
    
    if ( (iRow < nPoints) && (iCol < nTransforms) ) {
                
        // point
        float x_frame1 = xyzVecMat[iRow*3 + 0];
        float y_frame1 = xyzVecMat[iRow*3 + 1];
        float z_frame1 = xyzVecMat[iRow*3 + 2]; 
            
        // transform point
        float x_frame2 = rotationMat[iCol*9 + 0] * x_frame1 + rotationMat[iCol*9 + 1] * y_frame1 + rotationMat[iCol*9 + 2] * z_frame1 + translationVec[iCol*3 + 0];
        float y_frame2 = rotationMat[iCol*9 + 3] * x_frame1 + rotationMat[iCol*9 + 4] * y_frame1 + rotationMat[iCol*9 + 5] * z_frame1 + translationVec[iCol*3 + 1];
        float z_frame2 = rotationMat[iCol*9 + 6] * x_frame1 + rotationMat[iCol*9 + 7] * y_frame1 + rotationMat[iCol*9 + 8] * z_frame1 + translationVec[iCol*3 + 2];
  
        // compute point new pixel x,y coordinates in transformed frame
        if (z_frame1 > 10) { 
            float xPixel = x_frame2 / z_frame2;
            float yPixel = y_frame2 / z_frame2;
            
            int pixelColId = __float2int_rn(xPixel * fx + ppx);
            int pixelRowId = __float2int_rn(yPixel * fy + ppy);
            
            float zMeas_frame2 = zMat[pixelRowId * imgWidth + pixelColId];
            float zErr = zMeas_frame2 - z_frame2;
            
            if( (pixelRowId >= 0) && 
                (pixelRowId < imgHeight) && 
                (pixelColId >= 0) && 
                (pixelColId < imgWidth) &&
                (zMeas_frame2 > 10) &&
                (fabsf(zErr) < 100) ) {
                    
                zErrorMat[iRow*nTransforms + iCol] = zErr;
            }
        }
    }
}
''', 'multipleTransform_xyzPoints')

#------------------------------------------------------------------------------
  
class image_transform_class:
    def __init__(self, imgHeight, imgWidth, nPoi): 
        
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.nPoi = nPoi
    
        # currenty these are hardcoded for 480 x 848
        self.ppx = 423.337
        self.ppy = 238.688
        self.fx = 421.225
        self.fy = 421.225
        
        # for finite differences in computing jacobian
        self.deltaPos = 8.0     # 1 cm
        self.deltaAng = 3*0.003490 # 0.2 deg

        self.nTransforms = 13  # number of transforms 
        self.zErrorPerturbMat = cp.zeros((nPoi, self.nTransforms), dtype = cp.float32) # 6DOF, positive and negative
        self.jacobianMat = cp.zeros((nPoi, 6), dtype = cp.float32) # 6DOF, positive and negative
        self.zErrorVec = cp.zeros(nPoi)
        
        self.dx = cp.array([self.deltaPos, self.deltaPos, self.deltaPos, self.deltaAng, self.deltaAng, self.deltaAng])
        self.posIdx = np.array([0, 2, 4, 6, 8, 10], dtype = int)
        self.negIdx = np.array([1, 3, 5, 7, 9, 11], dtype = int)
        self.nullIdx = 12
        
        
    #--------------------------------------------------------------------------
    def transformXYZVecMat(self, \
                           xyzVecMat_transformed, \
                           pixelIdVec, \
                           zErrorVec, \
                           xyzVecMat,  \
                           zMat, \
                           Dcm,        \
                           translationVec):
        
        # point of interest matrix is shape [nFeatures*scale, scale]
        xyzVecMat_transformed *= 0
        pixelIdVec *= 0
        zErrorVec *= 0
    
        nPoints, _ = xyzVecMat.shape
        
        # Grid and block sizes
        block = (16,)
        grid = ( np.ceil(nPoints/block[0]).astype(int),  )
    
        # Call kernel
        cp_transform_xyzPoints(grid, block,  \
                   (xyzVecMat_transformed,
                    pixelIdVec, 
                    zErrorVec,
                    xyzVecMat,
                    zMat,
                    Dcm,
                    translationVec,
                    cp.int32(self.imgHeight),
                    cp.int32(self.imgWidth),
                    cp.float32(self.ppx),
                    cp.float32(self.ppy),
                    cp.float32(self.fx),
                    cp.float32(self.fy),
                    cp.int32(nPoints)) )  
            
            
    #--------------------------------------------------------------------------
    def multiTransformXYZVecMat(self, \
                               zErrorMat,  \
                               xyzVecMat,  \
                               zMat,       \
                               multiDcm,   \
                               multiTranslationVec):
        
        # point of interest matrix is shape [nFeatures*scale, scale]
        zErrorMat *= 0
    
        nPoints, _ = xyzVecMat.shape
        nTransforms, _ = multiTranslationVec.shape
        print(nPoints)
        print(nTransforms)
        # Grid and block sizes
        block = (8, 8)
        grid = ( np.ceil(nPoints/block[0]).astype(int), np.ceil(nTransforms/block[1]).astype(int) )
    
        # Call kernel
        cp_multiTransform_xyzPoints(grid, block,  \
                   (self.zErrorPerturbMat,
                    xyzVecMat,
                    zMat,
                    multiDcm,
                    multiTranslationVec,
                    cp.int32(self.imgHeight),
                    cp.int32(self.imgWidth),
                    cp.float32(self.ppx),
                    cp.float32(self.ppy),
                    cp.float32(self.fx),
                    cp.float32(self.fy),
                    cp.int32(nPoints),
                    cp.int32(nTransforms)) )  


    #--------------------------------------------------------------------------
    def computeTransformJacobian(self, \
                                xyzVecMat,   \
                                zMat,        \
                                Dcm,    \
                                translationVec):
        
        transPerturb_cpu, rotPerturb_cpu = createPerurbedTransforms(self.deltaAng, self.deltaPos, translationVec, Dcm)
        
        transPerturb = cp.array(transPerturb_cpu, dtype = cp.float32)
        rotPerturb = cp.array(rotPerturb_cpu, dtype = cp.float32)
        
        # point of interest matrix is shape [nFeatures*scale, scale]
        self.zErrorPerturbMat *= 0
    
        # Grid and block sizes
        block = (8, 8)
        grid = ( np.ceil(self.nPoi/block[0]).astype(int), np.ceil(self.nTransforms/block[1]).astype(int) )
    
        # Call kernel
        cp_multiTransform_xyzPoints(grid, block,  \
                   (self.zErrorPerturbMat,
                    xyzVecMat,
                    zMat,
                    rotPerturb,
                    transPerturb,
                    cp.int32(self.imgHeight),
                    cp.int32(self.imgWidth),
                    cp.float32(self.ppx),
                    cp.float32(self.ppy),
                    cp.float32(self.fx),
                    cp.float32(self.fy),
                    cp.int32(self.nPoi),
                    cp.int32(self.nTransforms)) )   
            
        self.jacobianMat = (self.zErrorPerturbMat[:, self.posIdx] - self.zErrorPerturbMat[:, self.negIdx]) / cp.tile(self.dx, (self.nPoi, 1))
        
        self.zErrorVec = self.zErrorPerturbMat[:, self.nullIdx]
            
            
#------------------------------------------------------------------------------
# helper functions   
def createPerurbedTransforms(deltaAng, deltaPos, translationVec0 = np.zeros(3), Dcm0 = np.eye(3)):
    
    # translation
    transPerturb = np.zeros((6,3))
    rotPerturb = np.zeros((6,9))
    for ii in range(3):
        
        # translation
        deltaVec = np.zeros((3))
        deltaVec[ii] = deltaPos
        transPerturb[ii*2]   = translationVec0[None,:] + deltaVec
        transPerturb[ii*2+1] = translationVec0[None,:] - deltaVec
        
        # rotation 
        angVec = np.zeros((3))
        angVec[ii] = deltaAng        

        Dcm_pos = eulerAngToDcm(angVec) @ Dcm0
        Dcm_neg = eulerAngToDcm(-angVec) @ Dcm0
        
        rotPerturb[ii*2]   = Dcm_pos.flatten()
        rotPerturb[ii*2+1] = Dcm_neg.flatten()

    transPerturb = np.concatenate( ( transPerturb,          
                                     np.tile(translationVec0, (6,1)),  
                                     translationVec0[None,:] ) )
    
    rotPerturb   = np.concatenate( ( np.tile(Dcm0.flatten(), (6,1)),  
                                     rotPerturb,                    
                                     Dcm0.flatten()[None,:] ) )
    
    return transPerturb, rotPerturb
    
    
def eulerAngToDcm(angVec):
    
    rot1 = np.array([ [1, 0, 0], [0, np.cos(angVec[0]), np.sin(angVec[0])], [0, -np.sin(angVec[0]), np.cos(angVec[0])] ])
    
    rot2 = np.array([ [np.cos(angVec[1]), 0, -np.sin(angVec[1])], [0, 1, 0], [ np.sin(angVec[1]), 0, np.cos(angVec[1])] ])
    
    rot3 = np.array([ [np.cos(angVec[2]), np.sin(angVec[2]), 0], [-np.sin(angVec[2]), np.cos(angVec[2]), 0], [0, 0, 1] ])
    
    return rot1 @ rot2 @ rot3
    

    
                 

        
