import numpy as np
import cupy as cp

#------------------------------------------------------------------------------
# cuda kernels
#------------------------------------------------------------------------------
cp_costFromFeatMatrix = cp.RawKernel(r'''
extern "C" __global__
void costMatrix(float* costMat,     // outputs
                float* featMat11,   // inputs
                float* featMat21,
                float* featMat31, 
                bool*  featMaskMat1,
                float* featMat12, 
                float* featMat22, 
                float* featMat32, 
                bool*  featMaskMat2,
                int costHeight, 
                int costWidth, 
                int featLength) {
 
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
    int img1_feature_index = iRow;
    int img2_feature_index = iCol;
    float nFeatures = 0.0;
    float featSum = 0.0;
    float e1 = 0.0;
    float e2 = 0.0;
    float e3 = 0.0;
    
    if( (iRow < costHeight) && (iCol < costWidth) ) {
        
        for (int f = 0; f < featLength; f++) {
        
            if (featMaskMat1[iRow*featLength + f] && featMaskMat2[iCol*featLength + f]) {
                
                e1 = featMat11[iRow*featLength + f] - featMat12[iCol*featLength + f];
                e2 = featMat21[iRow*featLength + f] - featMat22[iCol*featLength + f];
                e3 = featMat31[iRow*featLength + f] - featMat32[iCol*featLength + f];
                
                featSum += e1*e1 + e2*e2 + e3*e3;
                nFeatures += 1.0;
            }
        }
                
        if (nFeatures > 0.0) {   
            costMat[iRow*costWidth + iCol] = featSum / nFeatures;
        }
    }
}
''', 'costMatrix')


#------------------------------------------------------------------------------
cp_featMatrix = cp.RawKernel(r'''
extern "C" __global__
void featMatrix(float* featMat1,  // outputs
               float* featMat2,
               float* featMat3,
               bool*  featMaskMat,
               float* imgMat1,   // inputs
               float* imgMat2,
               float* imgMat3,
               bool* maskMat,
               int* imgPixelId_xVec, 
               int* imgPixelId_yVec,   
               int featHeight, 
               int featWidth, 
               int imgHeight,
               int imgWidth,
               int scale,
               int offset) {       
 
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    if( (iRow < featHeight) && (iCol < featWidth) ) {
        
        int img_feature_index = iRow;
        int img_xOffset = iCol / scale;
        int img_yOffset = iCol % scale;
        
        int img_x = imgPixelId_xVec[img_feature_index] + img_xOffset - offset;
        int img_y = imgPixelId_yVec[img_feature_index] + img_yOffset - offset;
        
        float rCenter = imgMat1[imgPixelId_xVec[iRow]*imgWidth + imgPixelId_yVec[iRow]];
        float gCenter = imgMat2[imgPixelId_xVec[iRow]*imgWidth + imgPixelId_yVec[iRow]];
        float bCenter = imgMat3[imgPixelId_xVec[iRow]*imgWidth + imgPixelId_yVec[iRow]];
        
        if ( (img_x >= 0) && (img_x < imgHeight) && (img_y >= 0) && (img_y < imgWidth) ) { 
                
            featMat1[iRow*featWidth + iCol]    = imgMat1[img_x*imgWidth + img_y];
            featMat2[iRow*featWidth + iCol]    = imgMat2[img_x*imgWidth + img_y];
            featMat3[iRow*featWidth + iCol]    = imgMat3[img_x*imgWidth + img_y];
            featMaskMat[iRow*featWidth + iCol] = maskMat[img_x*imgWidth + img_y];
        }
    }
}
''', 'featMatrix')


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
# main class
#------------------------------------------------------------------------------
class feature_matching_class:

    def __init__(self, imHeight, imgWidth, nFeatures, scale): 
        
        # parameters
        self.imgHeight = imHeight
        self.imgWidth = imgWidth
        self.nFeatures = nFeatures
        self.scale = scale
        self.offset = np.int32(1 + scale/2)
        self.featLength = scale*scale
        self.costHeight = nFeatures 
        self.costWidth = nFeatures 
        
        # working variables
        self.im1_featMats = self.generateFeatureMats(self.nFeatures, self.featLength)
        self.im2_featMats = self.generateFeatureMats(self.nFeatures, self.featLength)
        self.costMat = cp.zeros((self.nFeatures, self.nFeatures), dtype = cp.float32)
        
        
    def generateFeatureMats(self, nFeatures, featLength):
        featMats = {
            'feat_rMat':    cp.zeros((nFeatures, featLength), dtype = cp.float32),
            'feat_gMat':    cp.zeros((nFeatures, featLength), dtype = cp.float32),
            'feat_bMat':    cp.zeros((nFeatures, featLength), dtype = cp.float32),
            'feat_maskMat': cp.zeros((nFeatures, featLength), dtype = bool),
            'imgFeatureIdx': cp.zeros((2, nFeatures), dtype = cp.int32)
        }
        return featMats
        
    def createFeatureMat(self, featMatDict, rMat, gMat, bMat, maskMat, imgFeatureIdx):
        
        featMatDict['imgFeatureIdx'] = imgFeatureIdx
        
        generateFeaturetMat(featMatDict['feat_rMat'], \
                            featMatDict['feat_gMat'], \
                            featMatDict['feat_bMat'], \
                            featMatDict['feat_maskMat'], \
                            rMat, \
                            gMat, \
                            bMat, \
                            maskMat, \
                            imgFeatureIdx, \
                            self.scale,
                            self.offset)
            
            
    def set_img1_features(self, rMat, gMat, bMat, maskMat, imgFeatureIdx):
        self.createFeatureMat(self.im1_featMats, rMat, gMat, bMat, maskMat, imgFeatureIdx)
        
        
    def set_img2_features(self, rMat, gMat, bMat, maskMat, imgFeatureIdx):
        self.createFeatureMat(self.im2_featMats, rMat, gMat, bMat, maskMat, imgFeatureIdx)       
          
        
    def computeFeatureMatches(self):
            
        self.createCostMat( self.im1_featMats['feat_rMat'], \
                            self.im1_featMats['feat_gMat'], \
                            self.im1_featMats['feat_bMat'], \
                            self.im1_featMats['feat_maskMat'], \
                            self.im2_featMats['feat_rMat'], \
                            self.im2_featMats['feat_gMat'], \
                            self.im2_featMats['feat_bMat'], \
                            self.im2_featMats['feat_maskMat'])

        costMat = self.costMat.get()

        colMin = costMat.argmin(axis = 0)
        rowMin = costMat.argmin(axis = 1)

        idxMatch = rowMin[colMin] == np.arange(self.nFeatures)

        idxMatchImg2 = np.arange(self.nFeatures)[idxMatch]
        idxMatchImg1 = colMin[idxMatchImg2]
        
        cornerMatchedIdx1 = self.im1_featMats['imgFeatureIdx'][:, idxMatchImg1]
        cornerMatchedIdx2 = self.im2_featMats['imgFeatureIdx'][:, idxMatchImg2]
        
        return cornerMatchedIdx1, cornerMatchedIdx2

        
    def createCostMat(self, 
                      featMat11, 
                      featMat21, 
                      featMat31, 
                      featMaskMat1, 
                      featMat12, 
                      featMat22, 
                      featMat32, 
                      featMaskMat2):
        
        self.costMat *= 0
        
        # fine cost matrix
        block = (8, 8)
        grid = (int(self.costWidth/block[0]), int(self.costHeight/block[1]))
    
        # call kernel
        cp_costFromFeatMatrix(grid, block,  \
                   (self.costMat,  
                    featMat11,    
                    featMat21, 
                    featMat31, 
                    featMaskMat1,
                    featMat12, 
                    featMat22, 
                    featMat32, 
                    featMaskMat2,
                    self.costHeight,
                    self.costWidth,
                    self.featLength) )
            
 
            
#------------------------------------------------------------------------------            
#  helper functions
#------------------------------------------------------------------------------


def generateFeaturetMat(featMat1, \
                        featMat2, \
                        featMat3, \
                        featMaskMat, \
                        imgMat1, \
                        imgMat2, \
                        imgMat3, \
                        maskMat, \
                        imgPixelIdx, \
                        scale,
                        offset):
    
    # point of interest matrix is shape [nFeatures*scale, scale]
    featMat1 *= 0
    featMat2 *= 0
    featMat3 *= 0

    featHeight, featWidth = featMat1.shape
    imgHeight, imgWidth = imgMat1.shape
    
    # Grid and block sizes
    block = (8, 8)
    grid = ( np.ceil(featWidth/block[0]).astype(int), np.ceil(featHeight/block[1]).astype(int) )

    # Call kernel
    cp_featMatrix(grid, block,  \
               (featMat1, 
                featMat2, 
                featMat3, 
                featMaskMat,
                imgMat1,
                imgMat2,
                imgMat3,
                maskMat,
                imgPixelIdx[0,:], 
                imgPixelIdx[1,:], 
                featHeight,
                featWidth,
                imgHeight,
                imgWidth,
                scale,
                offset) )  
        
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