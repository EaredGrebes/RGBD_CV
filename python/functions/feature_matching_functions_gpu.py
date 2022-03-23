import numpy as np
import cupy as cp

#------------------------------------------------------------------------------
# cuda kernels
cp_costMatrix = cp.RawKernel(r'''
extern "C" __global__
void costMatrix(float* costMat,   // outputs
                float* imgMat1,   // inputs
                float* imgMat2, 
                bool* maskMat1,
                bool* maskMat2,
                int* img1Id_xVec, 
                int* img1Id_yVec,   
                int* img2Id_xVec,
                int* img2Id_yVec,
                int costHeight, 
                int costWidth, 
                int imgHeight,
                int imgWidth,
                int scale,
                int offset) {
 
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    if( (iRow < costHeight) && (iCol < costWidth) ) {
        
        int img1_feature_index = iRow / scale;
        int img2_feature_index = iCol / scale;
        
        int img_xOffset = iRow - (img1_feature_index * scale);
        int img_yOffset = iCol - (img2_feature_index * scale);
        
        int img1_x = img1Id_xVec[img1_feature_index] + img_xOffset - offset;
        int img1_y = img1Id_yVec[img1_feature_index] + img_yOffset - offset;
        
        int img2_x = img2Id_xVec[img2_feature_index] + img_xOffset - offset;
        int img2_y = img2Id_yVec[img2_feature_index] + img_yOffset - offset;
        
        if ( (img1_x >= 0) && (img1_x < imgHeight) && (img2_y >= 0) && (img2_y < imgWidth) ) {
                    
            if (maskMat1[img1_x*imgWidth + img1_y] == true) && (maskMat2[img2_x*imgWidth + img2_y] == true){
                
                float err = imgMat1[img1_x*imgWidth + img1_y] - imgMat2[img2_x*imgWidth + img2_y]
                costMat[iRow*costWidth + iCol] = err * err
            }     
        }
    }
}
''', 'costMatrix')


#------------------------------------------------------------------------------
  
class feature_matching_class:

    def __init__(self, imHeight, imgWidth, nFeatures, scale): 
        
        self.imgHeight = imHeight
        self.imgWidth = imgWidth
    
        self.nFeatures = nFeatures
        self.scale = scale
        self.offset = np.int32(1 + scale/2)
        self.costHeight = nFeatures * scale 
        self.costWidth = nFeatures * scale 
        
        self.costMat = cp.zeros((self.costHeight, self.costWidth), dtype = cp.float32)
        
        
    def createCostMatrix(self, imgMat1, imgMat2, maskMat1, maskMat2, img1_pixelIdx, img2_pixelIdx):
        
        self.costMat *= 0
        
        # Grid and block sizes
        block = (8, 8)
        grid = (int(self.costWidth/block[0]), int(self.costHeight/block[1]))
    
        # Call kernel
        cp_costMatrix(grid, block,  \
                   (self.costMat,   
                    imgMat1,    
                    imgMat2, 
                    maskMat1,
                    maskMat2,
                    img1_pixelIdx[0,:], 
                    img1_pixelIdx[1,:],   
                    img2_pixelIdx[0,:],
                    img2_pixelIdx[1,:],
                    cp.int32(self.height), 
                    cp.int32(self.width), 
                    cp.int32(self.imgHeight),
                    cp.int32(self.imgWidth),
                    cp.int32(self.scale),
                    cp.int32(self.offset)) )
                

    def createMatchMatrix(self, rgbMat1, maskMat1, rgbMat2, masMat2):
        
        tmp3 = 5;