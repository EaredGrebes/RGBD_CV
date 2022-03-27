import numpy as np
import cupy as cp
                   
#------------------------------------------------------------------------------
cp_transform_xyzPoints = cp.RawKernel(r'''
extern "C" __global__
void transform_xyzPoints(int* pixelIdVec,     // outputs
                         float* xyzVecMat,    // inputs
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
  
        // compute point new pixel x,y coordinates in transformed frame
        float xPixel = x2 / z2;
        float yPixel = y2 / z2;
        
        int pixelColId = __float2int_rn(xPixel * fx + ppx);
        int pixelRowId = __float2int_rn(yPixel * fy + ppy);
        
        if( (pixelRowId >= 0) && (pixelRowId < imgHeight) && (pixelColId >= 0) && (pixelColId < imgWidth) ) {
                
            pixelIdVec[iRow*2 + 0] = pixelRowId;
            pixelIdVec[iRow*2 + 1] = pixelColId;
        }
    }
}
''', 'transform_xyzPoints')


#------------------------------------------------------------------------------
  
class image_transform_class:
    # currently only works for 480 X 848 realsense camera
    def __init__(self): 
        
        self.imgHeight = 480
        self.imgWidth = 848
    
        self.ppx = 423.337
        self.ppy = 238.688
        self.fx = 421.225
        self.fy = 421.225
        

    def transformXYZVecMat(self, \
                           pixelIdVec, \
                           xyzVecMat,  \
                           Dcm,        \
                           translationVec):
        
        # point of interest matrix is shape [nFeatures*scale, scale]
        pixelIdVec *= 0
    
        nPoints, _ = xyzVecMat.shape
        
        # Grid and block sizes
        block = (16,)
        grid = ( np.ceil(nPoints/block[0]).astype(int),  )
    
        # Call kernel
        cp_transform_xyzPoints(grid, block,  \
                   (pixelIdVec, 
                    xyzVecMat,
                    Dcm,
                    translationVec,
                    cp.int32(self.imgHeight),
                    cp.int32(self.imgWidth),
                    cp.float32(self.ppx),
                    cp.float32(self.ppy),
                    cp.float32(self.fx),
                    cp.float32(self.fy),
                    cp.int32(nPoints)) )  

        
