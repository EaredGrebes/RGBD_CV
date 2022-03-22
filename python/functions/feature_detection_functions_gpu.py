import numpy as np
import cupy as cp

#------------------------------------------------------------------------------
# cuda kernels
cp_gradient = cp.RawKernel(r'''
extern "C" __global__
void computeGradient(float* gradxMat,  // outputs
                     float* gradyMat, 
                     float* imgMat,    // inputs
                     bool* maskMat,       
                     int* pixelOffset_x, 
                     int* pixelOffset_y,
                     float* B_row1,
                     float* B_row2,
                     int height, 
                     int width, 
                     int nP) {
  
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    int buf = 3;
    if( (iRow >= buf) && (iRow < height-buf) && (iCol >= buf) && (iCol < width-buf) ) {

        if (maskMat[iRow*width + iCol] == true) {
            
            float gradx = 0;
            float grady = 0;
            int maskCount = 0;
            int iIm = 0;
            int jIm = 0;
            
            float val0 = imgMat[iRow*width + iCol];
            
            for (int ii = 0; ii < nP; ii++) {
                   
                iIm = iRow + pixelOffset_x[ii];
                jIm = iCol + pixelOffset_y[ii];
                
                if (maskMat[iIm*width + jIm] == true) {
                    maskCount = maskCount + 1;
                    
                    gradx = gradx + B_row1[ii] * (imgMat[iIm*width + jIm] - val0);
                    grady = grady + B_row2[ii] * (imgMat[iIm*width + jIm] - val0);        
                }
            }
            
            if (maskCount == nP){
               gradxMat[iRow*width + iCol] = gradx;
               gradyMat[iRow*width + iCol] = grady;
            }
        }
    }
}
''', 'computeGradient')

#------------------------------------------------------------------------------
cp_crossProdMat = cp.RawKernel(r'''
extern "C" __global__
void computeCrossProdMat(float* crossProdMat,  // outputs
                         float* gradxMat,      // inputs
                         float* gradyMat,
                         int height, 
                         int width) {
  
    int jj = blockIdx.x * blockDim.x  + threadIdx.x;
    int ii = blockIdx.y * blockDim.y  + threadIdx.y;
  
    int buf = 3;
    int l = 2;
    
    if( (ii >= buf) && (ii < height-buf) && (jj >= buf) && (jj < width-buf) ) {
            
        float gradx = 1 * (gradxMat[(ii-l)*width + jj-l] * gradyMat[(ii+l)*width + jj-l] - gradyMat[(ii-l)*width + jj-l] * gradxMat[(ii+l)*width + jj-l]) + 
                      2 * (gradxMat[(ii-l)*width + jj  ] * gradyMat[(ii+l)*width + jj  ] - gradyMat[(ii-l)*width + jj  ] * gradxMat[(ii+l)*width + jj  ]) + 
                      1 * (gradxMat[(ii-l)*width + jj+l] * gradyMat[(ii+l)*width + jj+l] - gradyMat[(ii-l)*width + jj-l] * gradxMat[(ii+l)*width + jj-l]);
        
        
        float grady = 1 * (gradxMat[(ii-l)*width + jj-l] * gradyMat[(ii-l)*width + jj+l] - gradyMat[(ii-l)*width + jj-l] * gradxMat[(ii-l)*width + jj+l]) + 
                      2 * (gradxMat[ii*width     + jj-l] * gradyMat[ii*width     + jj+l] - gradyMat[ii*width     + jj-l] * gradxMat[ii*width     + jj+l]) + 
                      1 * (gradxMat[(ii+l)*width + jj-l] * gradyMat[(ii+l)*width + jj+l] - gradyMat[(ii+l)*width + jj-l] * gradxMat[(ii+l)*width + jj+l]);    
           
        crossProdMat[ii*width + jj] = sqrtf(gradx*gradx + grady*grady); 
    }
}
''', 'computeCrossProdMat')


#------------------------------------------------------------------------------
cp_findLocalMax = cp.RawKernel(r'''
extern "C" __global__
void findLocalMax(float* localMaxMat,  // outputs
                  float* courseMaxVec,
                  int* pixelXLocVec,
                  int* pixelYLocVec,
                  float* mat,          // inputs
                  int c,
                  int height, 
                  int width,
                  int height_c,
                  int width_c) {
  
    int jj = blockIdx.x * blockDim.x  + threadIdx.x;
    int ii = blockIdx.y * blockDim.y  + threadIdx.y;
  
    
    if( (ii >= c) && (ii < height-c) && (jj >= c) && (jj < width-c) ) {
        
        float maxVal = 0;
        for (int iic = -c; iic <= c; iic++) {
            for (int jjc = -c; jjc <= c; jjc++) {
           
                maxVal = fmaxf( mat[(ii+iic)*width + jj+jjc], maxVal );
            }
        }
        // pixel is a local maximum
       if (mat[ii*width + jj] >= maxVal) {
               
           localMaxMat[ii*width + jj] = mat[ii*width + jj];
           
           int ii_course = __float2int_rd( __int2float_rn(ii) / __int2float_rn(c) );
           int jj_course = __float2int_rd( __int2float_rn(jj) / __int2float_rn(c) );
           
           courseMaxVec[ii_course * width_c + jj_course] = mat[ii*width + jj];
           pixelXLocVec[ii_course * width_c + jj_course] = ii;
           pixelYLocVec[ii_course * width_c + jj_course] = jj;
       } 
    }
}
''', 'findLocalMax')


#------------------------------------------------------------------------------
  
class corner_detector_class:

    def __init__(self): 
        
        nP = 16
        offsetMat = np.array([[-3, -3, -2, -1,  0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3], \
                              [ 0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3, -3, -3, -2, -1]])            
        
        # create least squares matrix for computing quadratic model of cost function from offset error points
        A = np.zeros((nP, 5))
        for offset in range(nP):
            x = offsetMat[0, offset]
            y = offsetMat[1, offset]
            
            A[offset, :] = np.array([x, y, x*y, x*x, y*y])
        
        B = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        
        self.pixelOffsetMat = cp.array(offsetMat, dtype = cp.int32, copy=False)
        self.B = cp.array(B, dtype = cp.float32, copy=False)
        
        
    def findCornerPoints(self, greyMat, maskMat, nMax):
        # greyMat and maskMat must be cupy arrays
        return findCornerPoints(greyMat, maskMat, self.pixelOffsetMat, self.B, nMax)


#------------------------------------------------------------------------------
def computeGradientMat(gradxMat, gradyMat, pixelOffsetMat, B, imgMat, maskMat, height, width):
    
    nP = pixelOffsetMat.shape[1]
    
    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_gradient(grid, block,  \
               (gradxMat,          # outputs
                gradyMat,
                imgMat,            # inputs
                maskMat,
                pixelOffsetMat[0,:], 
                pixelOffsetMat[1,:], 
                B[0,:],
                B[1,:],
                cp.int32(height), 
                cp.int32(width),
                cp.int32(nP)) )
                

#------------------------------------------------------------------------------
def computeCrossProdMat(crossProdMat, gradxMat, gradyMat, height, width):

    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_crossProdMat(grid, block,  \
               (crossProdMat,   # outputs
                gradxMat,       # inputs
                gradyMat,
                cp.int32(height), 
                cp.int32(width)) )


#------------------------------------------------------------------------------
def findLocalMax(localMaxMat, courseMaxVec, pixelXVec, pixelYVec, mat, localScale, height, width):
    
    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_findLocalMax(grid, block,  \
               (localMaxMat,   # outputs
                courseMaxVec,
                pixelXVec,
                pixelYVec,
                mat,            # inputs
                cp.int32(localScale),
                cp.int32(height), 
                cp.int32(width),
                cp.int32(height / localScale), 
                cp.int32(width / localScale)) )     
    
    
#------------------------------------------------------------------------------
def findCornerPoints(gradxMat, \
                     gradyMat, \
                     crossProdMat, \
                     coarseMaxMat, \
                     courseMaxVec, \
                     pixelXVec, \
                     pixelYVec, \
                     greyMat, \
                     maskMat, \
                     pixelOffsetMat, 
                     B, 
                     nMax, 
                     c, 
                     height, 
                     width):
    
    computeGradientMat(gradxMat,  \
                            gradyMat,   \
                            pixelOffsetMat, \
                            B,              \
                            greyMat,    \
                            maskMat,    \
                            height,         \
                            width)
        
    computeCrossProdMat(crossProdMat, \
                              gradxMat, \
                              gradyMat, \
                              height, \
                              width)   
    
    findLocalMax(coarseMaxMat,  \
                       courseMaxVec,  \
                       pixelXVec,     \
                       pixelYVec,     \
                       crossProdMat,  \
                       c,      \
                       height, \
                       width)
                    
    return 


#--------------------------------------------------------------------------
def computeMatError(mat1, mat2, mask):
    matErr = mat1 - mat2
    f = np.sum(matErr[mask] * matErr[mask]) / np.sum(mask).astype(float)
    #f = np.sum(np.abs(matErr))
    return f


#------------------------------------------------------------------------------
def computeMatchCost(rgb1Mat, rgb2Mat, mask1Mat, mask2Mat, id2dMax1, id2dMax2):
    
    l = 7
    maskMat1 = mask1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1]
    maskMat2 = mask2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1]
    mask = np.logical_and(maskMat1, maskMat2)
    
    fr = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1, 0], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1, 0], mask)
    fg = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1, 1], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1, 1], mask)
    fb = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1, 2], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1, 2], mask)
    
    f = fr + fg + fb
    return f

