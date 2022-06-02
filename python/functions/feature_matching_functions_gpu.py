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
                
                //featSum += e1*e1 + e2*e2 + e3*e3;
                featSum += fabs(e1) + fabs(e2) + fabs(e3);
                nFeatures += 1.0;
            }
        }
                
        if (nFeatures > 0.0) {   
            costMat[iRow*costWidth + iCol] = featSum / nFeatures;
        } else {
            costMat[iRow*costWidth + iCol] = 666.6;
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
        
        //float rCenter = imgMat1[imgPixelId_xVec[iRow]*imgWidth + imgPixelId_yVec[iRow]];
        //float gCenter = imgMat2[imgPixelId_xVec[iRow]*imgWidth + imgPixelId_yVec[iRow]];
        //float bCenter = imgMat3[imgPixelId_xVec[iRow]*imgWidth + imgPixelId_yVec[iRow]];
        
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
# main class
#------------------------------------------------------------------------------
class feature_matching_class:

    def __init__(self, imHeight, imgWidth, nFeatures1, nFeatures2, scale): 
        
        # parameters
        self.imgHeight = imHeight
        self.imgWidth = imgWidth
        self.nFeatures1 = nFeatures1
        self.nFeatures2 = nFeatures2
        self.scale = scale
        #self.offset = np.int32(1 + scale/2)
        self.offset = np.int32((scale-1)/2)
        self.featLength = scale*scale
        self.costHeight = nFeatures1
        self.costWidth = nFeatures2 
        self.dist_thresh2 = 40**2
        
        # working variables
        self.im1_featMats = self.generateFeatureMats(self.nFeatures1, self.featLength)
        self.im2_featMats = self.generateFeatureMats(self.nFeatures2, self.featLength)
        self.costMat = cp.zeros((self.costHeight, self.costWidth), dtype = cp.float32)
        
        
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

        idxMatch = rowMin[colMin] == np.arange(self.nFeatures1)

        idxMatchImg2 = np.arange(self.nFeatures2)[idxMatch]
        idxMatchImg1 = colMin[idxMatchImg2]
        
        cornerMatchedIdx1 = self.im1_featMats['imgFeatureIdx'][:, idxMatchImg1]
        cornerMatchedIdx2 = self.im2_featMats['imgFeatureIdx'][:, idxMatchImg2]
        
        # remove points whos matched pixel distances are large
        err = cornerMatchedIdx1 - cornerMatchedIdx2
        matchPixelDist = np.sum(err * err, axis = 0)

        cornerMatchedIdx1 = cornerMatchedIdx1[:,matchPixelDist < self.dist_thresh2]
        cornerMatchedIdx2 = cornerMatchedIdx2[:,matchPixelDist < self.dist_thresh2]
        
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
        grid = ( np.ceil(self.costWidth/block[0]).astype(int), np.ceil(self.costHeight/block[1]).astype(int) )
        
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
         