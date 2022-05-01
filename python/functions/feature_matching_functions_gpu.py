import numpy as np
import cupy as cp

#------------------------------------------------------------------------------
cp_sumCostMatrix = cp.RawKernel(r'''
extern "C" __global__
void sumCostMatrix( float* costCoarseMat,   // outputs
                    float* costFineMat,     // inputs
                    int coarseHeight, 
                    int coarseWidth, 
                    int fineHeight, 
                    int fineWidth,                    
                    int scale) {
 
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    if( (iRow < coarseHeight) && (iCol < coarseWidth) ) {
    
        int id_x1 = iRow * scale;
        int id_x2 = (iRow + 1) * scale;
        int id_y1 = iCol * scale;
        int id_y2 = (iCol + 1) * scale;
        
        float sum = 0;
        float numPoints = 0;
        
        for (int ii = id_x1; ii < id_x2; ii++) {
            for (int jj = id_y1; jj < id_y2; jj++) {
                    
                sum = sum + costFineMat[ii*fineWidth + jj];
                
                if (costFineMat[ii*fineWidth + jj] > 0){
                   numPoints = numPoints + 1;
                }
            }
        }
        if (numPoints > 0.0){
            costCoarseMat[iRow*coarseWidth + iCol] = sum / numPoints;
        }

    }
}
''', 'sumCostMatrix')


#------------------------------------------------------------------------------
cp_costFromPoiMatrix = cp.RawKernel(r'''
extern "C" __global__
void costMatrix(float* costMat,   // outputs
                float* poiMat11,   // inputs
                float* poiMat21,
                float* poiMat31, 
                bool*  poiMaskMat1,
                float* poiMat12, 
                float* poiMat22, 
                float* poiMat32, 
                bool*  poiMaskMat2,
                int costHeight, 
                int costWidth, 
                int scale) {
 
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    if( (iRow < costHeight) && (iCol < costWidth) ) {
        
        int img1_feature_index = iRow / scale;
        int img2_feature_index = iCol / scale;
        
        int img_xOffset = iRow - (img1_feature_index * scale);
        int img_yOffset = iCol - (img2_feature_index * scale);
        
        int poi1_x = (img1_feature_index * scale) + img_xOffset;
        int poi1_y = img_yOffset;
        
        int poi2_x = (img2_feature_index * scale) + img_xOffset;
        int poi2_y = img_yOffset;
                
        float err1 = poiMat11[poi1_x*scale + poi1_y] - poiMat12[poi2_x*scale + poi2_y];
        float err2 = poiMat21[poi1_x*scale + poi1_y] - poiMat22[poi2_x*scale + poi2_y];
        float err3 = poiMat31[poi1_x*scale + poi1_y] - poiMat32[poi2_x*scale + poi2_y];
        
        if (poiMaskMat1[poi1_x*scale + poi1_y] && poiMaskMat2[poi2_x*scale + poi2_y]) {
                
            costMat[iRow*costWidth + iCol] = err1*err1 + err2*err2 + err3*err3;
            
        } else {
            costMat[iRow*costWidth + iCol] = 0.0;
        }
    }
}
''', 'costMatrix')


#------------------------------------------------------------------------------
cp_poiMatrix = cp.RawKernel(r'''
extern "C" __global__
void poiMatrix(float* poiMat1,  // outputs
               float* poiMat2,
               float* poiMat3,
               bool*  poiMaskMat,
               float* imgMat1,   // inputs
               float* imgMat2,
               float* imgMat3,
               bool* maskMat,
               int* imgPixelId_xVec, 
               int* imgPixelId_yVec,   
               int poiHeight, 
               int poiWidth, 
               int imgHeight,
               int imgWidth,
               int scale,
               int offset) {
                    
 
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    if( (iRow < poiHeight) && (iCol < poiWidth) ) {
        
        int img_feature_index = iRow / scale;
        
        int img_xOffset = iRow - (img_feature_index * scale);
        int img_yOffset = iCol;
        
        int img_x = imgPixelId_xVec[img_feature_index] + img_xOffset - offset;
        int img_y = imgPixelId_yVec[img_feature_index] + img_yOffset - offset;
        
        if ( (img_x >= 0) && (img_x < imgHeight) && (img_y >= 0) && (img_y < imgWidth) ) { 
                
            poiMat1[iRow*poiWidth + iCol] = imgMat1[img_x*imgWidth + img_y];
            poiMat2[iRow*poiWidth + iCol] = imgMat2[img_x*imgWidth + img_y];
            poiMat3[iRow*poiWidth + iCol] = imgMat3[img_x*imgWidth + img_y];
            poiMaskMat[iRow*poiWidth + iCol] = maskMat[img_x*imgWidth + img_y];
        }
    }
}
''', 'poiMatrix')


#------------------------------------------------------------------------------
cp_xyzMatrix = cp.RawKernel(r'''
extern "C" __global__
void xyzMatrix(float* xyzVecMat,  // outputs
               float* greyVec,
               float* xMat,       // inputs
               float* yMat,
               float* zMat, 
               float* greyMat,
               bool*  maskMat,
               int* imgPixelId_xVec, 
               int* imgPixelId_yVec,   
               int xyzHeight, 
               int xyzWidth, 
               int imgHeight,
               int imgWidth,
               int scale,
               int scale2,
               int offset,
               int nPoints) {
                    
 
    int iRow = blockIdx.x * blockDim.x  + threadIdx.x;
  
    if (iRow < xyzHeight) {
        
        // every feature is composed of scale2 number of pixels
        int img_feature_index = iRow / scale2;
        
        if (img_feature_index < nPoints) {
        
            // this is the pixel offset index for a particular feature
            int img_Offset = iRow - (img_feature_index * scale2);
            
            // this is the pixel offset in image x,y coordinates
            int img_xOffset = img_Offset / scale;
            int img_yOffset = img_Offset - (img_xOffset * scale);
            
            int img_x = imgPixelId_xVec[img_feature_index] + img_xOffset - offset;
            int img_y = imgPixelId_yVec[img_feature_index] + img_yOffset - offset;
            
            if ( (img_x >= 0) && (img_x < imgHeight) && (img_y >= 0) && (img_y < imgWidth) && (maskMat[img_x*imgWidth + img_y]) ) { 
                    
                xyzVecMat[iRow*xyzWidth + 0] = xMat[img_x*imgWidth + img_y];
                xyzVecMat[iRow*xyzWidth + 1] = yMat[img_x*imgWidth + img_y];
                xyzVecMat[iRow*xyzWidth + 2] = zMat[img_x*imgWidth + img_y];
                
                greyVec[iRow] = greyMat[img_x*imgWidth + img_y];
            }
        }
    }
}
''', 'xyzMatrix')


#------------------------------------------------------------------------------
  
class feature_matching_class:

    def __init__(self, imHeight, imgWidth, nFeatures, scale): 
        
        self.imgHeight = imHeight
        self.imgWidth = imgWidth
    
        self.nFeatures = nFeatures
        self.scale = scale
        self.costFineHeight = nFeatures * scale 
        self.costFineWidth = nFeatures * scale 
        self.costCoarseHeight = nFeatures 
        self.costCoarseWidth = nFeatures 
        
        self.costFineMat = cp.zeros((self.costFineHeight, self.costFineHeight), dtype = cp.float32)
        self.maskFineMat = cp.zeros((self.costFineHeight, self.costFineHeight), dtype = cp.bool)
        self.costCoarseMat = cp.zeros((self.nFeatures, self.nFeatures), dtype = cp.float32)
        
        
    def computeMatches( self, \
                        poi_rMat1, \
                        poi_gMat1, \
                        poi_bMat1, \
                        poi_maskMat1,    \
                        cornerPointIdx1, \
                        poi_rMat2, \
                        poi_gMat2, \
                        poi_bMat2, \
                        poi_maskMat2,
                        cornerPointIdx2):
            
        self.createCostMat( poi_rMat1, \
                            poi_gMat1, \
                            poi_bMat1, \
                            poi_maskMat1, \
                            poi_rMat2, \
                            poi_gMat2, \
                            poi_bMat2, \
                            poi_maskMat2)

        costMat = self.costCoarseMat.get()

        colMin = costMat.argmin(axis = 0)
        rowMin = costMat.argmin(axis = 1)

        idxMatch = rowMin[colMin] == np.arange(self.nFeatures)

        idxMatchImg2 = np.arange(self.nFeatures)[idxMatch]
        idxMatchImg1 = colMin[idxMatchImg2]
        
        cornerMatchedIdx1 = cornerPointIdx1[:, idxMatchImg1]
        cornerMatchedIdx2 = cornerPointIdx2[:, idxMatchImg2]
        
        return cornerMatchedIdx1, cornerMatchedIdx2

        
    def createCostMat(self, poiMat11, poiMat21, poiMat31, poiMaskMat1, poiMat12, poiMat22, poiMat32, poiMaskMat2):
        
        self.costFineMat *= 0
        self.costCoarseMat *= 0
        
        # fine cost matrix
        block = (8, 8)
        grid = (int(self.costFineWidth/block[0]), int(self.costFineHeight/block[1]))
    
        # call kernel
        cp_costFromPoiMatrix(grid, block,  \
                   (self.costFineMat,  
                    poiMat11,    
                    poiMat21, 
                    poiMat31, 
                    poiMaskMat1,
                    poiMat12, 
                    poiMat22, 
                    poiMat32, 
                    poiMaskMat2,
                    cp.int32(self.costFineHeight),
                    cp.int32(self.costFineWidth),
                    cp.int32(self.scale)) )
            
        # use block averaging to compute coarse cost mat 
        block = (8, 8)
        grid = (int(self.nFeatures/block[0]), int(self.nFeatures/block[1]))
    
        # call kernel
        cp_sumCostMatrix(grid, block,  \
                   (self.costCoarseMat,   
                    self.costFineMat,    
                    cp.int32(self.costCoarseHeight),
                    cp.int32(self.costCoarseWidth),
                    cp.int32(self.costFineHeight),
                    cp.int32(self.costFineWidth),
                    cp.int32(self.scale)) )
 
            
#------------------------------------------------------------------------------            
# functions

# creates a tall rectangular matrix of vertically stacked squares
# each sqaure is a scale X scale box around a feature point coordinate from a 2d image
# poiMatrix shape: (nFeatures*scale) X scale
# done in batches of 3, for rgb, or xyz
def generatePointsOfInterestMat(poiMat1, \
                                poiMat2, \
                                poiMat3, \
                                poiMaskMat, \
                                imgMat1, \
                                imgMat2, \
                                imgMat3, \
                                maskMat, \
                                imgPixelIdx, \
                                scale,
                                offset):
    
    # point of interest matrix is shape [nFeatures*scale, scale]
    poiMat1 *= 0
    poiMat2 *= 0
    poiMat3 *= 0

    poiHeight, poiWidth = poiMat1.shape
    imgHeight, imgWidth = imgMat1.shape
    
    # Grid and block sizes
    block = (8, 8)
    grid = ( np.ceil(poiWidth/block[0]).astype(int), np.ceil(poiHeight/block[1]).astype(int) )

    # Call kernel
    cp_poiMatrix(grid, block,  \
               (poiMat1, 
                poiMat2, 
                poiMat3, 
                poiMaskMat,
                imgMat1,
                imgMat2,
                imgMat3,
                maskMat,
                imgPixelIdx[0,:], 
                imgPixelIdx[1,:], 
                cp.int32(poiHeight),
                cp.int32(poiWidth),
                cp.int32(imgHeight),
                cp.int32(imgWidth),
                cp.int32(scale),
                cp.int32(offset)) )  
        
# creates a tall rectangular matrix of vertically stacked [x,y,z] vector rows
# each vector is taken from a box around a set of feature point coordinate from a 2d image
def generateXYZVecMat(xyzVecMat, \
                      greyVec,   \
                      xMat, \
                      yMat, \
                      zMat, \
                      greyMat, \
                      maskMat, \
                      imgPixelIdx, \
                      scale,       \
                      offset):
    
    # point of interest matrix is shape [nFeatures*scale, scale]
    xyzVecMat *= 0

    xyzHeight, xyzWidth = xyzVecMat.shape
    imgHeight, imgWidth = xMat.shape
    nPoints = imgPixelIdx.shape[1]

    # Grid and block sizes
    block = (16,)
    grid = ( np.ceil(xyzHeight/block[0]).astype(int),  )

    # Call kernel
    cp_xyzMatrix(grid, block,  \
               (xyzVecMat, 
                greyVec,
                xMat, 
                yMat,
                zMat,
                greyMat,
                maskMat,
                imgPixelIdx[0,:], 
                imgPixelIdx[1,:], 
                cp.int32(xyzHeight),
                cp.int32(xyzWidth),
                cp.int32(imgHeight),
                cp.int32(imgWidth),
                cp.int32(scale),
                cp.int32(scale*scale),
                cp.int32(offset),
                cp.int32(nPoints)) )   