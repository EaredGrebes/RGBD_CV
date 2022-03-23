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
           
           //int ii_course = __float2int_rd( __int2float_rn(ii) / __int2float_rn(c) );
           //int jj_course = __float2int_rd( __int2float_rn(jj) / __int2float_rn(c) );
           
           int ii_course = ii / c;
           int jj_course = jj / c;
           
           courseMaxVec[ii_course * width_c + jj_course] = mat[ii*width + jj];
           pixelXLocVec[ii_course * width_c + jj_course] = ii;
           pixelYLocVec[ii_course * width_c + jj_course] = jj;
       } 
    }
}
''', 'findLocalMax')


#------------------------------------------------------------------------------
  
class corner_detector_class:

    def __init__(self, height, width, c, nMax): 
        
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
        
        # matricies used in the corner detection computation
        self.height = height
        self.width = width
        self.c = c
        self.nMax = nMax
        
        height_c = int(height / c)
        width_c = int(width / c)
        
        # working variables
        self.gradxMat = cp.zeros((height, width), dtype = cp.float32)
        self.gradyMat = cp.zeros((height, width), dtype = cp.float32)
        self.crossProdMat = cp.zeros((height, width), dtype = cp.float32)
        self.coarseMaxMat = cp.zeros((height, width), dtype = cp.float32)

        # outputs
        self.courseMaxVec = cp.zeros(height_c * width_c, dtype = cp.float32)
        self.courseMaxVec_cpu = np.zeros(height_c * width_c, dtype = np.float32)
        self.pixelXVec = cp.zeros(height_c * width_c, dtype = cp.int32)
        self.pixelYVec = cp.zeros(height_c * width_c, dtype = cp.int32)
        
        self.idx2dMax = cp.zeros((2,nMax), dtype = cp.int32)
        
    def findCornerPoints(self, greyMat, maskMat):
        
        # greyMat and maskMat must be cupy arrays
        findCornerPoints(self.gradxMat, \
                        self.gradyMat, \
                        self.crossProdMat, \
                        self.coarseMaxMat, \
                        self.courseMaxVec, \
                        self.pixelXVec, \
                        self.pixelYVec, \
                        greyMat, \
                        maskMat, \
                        self.pixelOffsetMat, 
                        self.B, 
                        self.nMax, 
                        self.c, 
                        self.height, 
                        self.width)
            
        self.courseMaxVec_cpu = self.courseMaxVec.get()
        idxMaxSorted = self.courseMaxVec_cpu.argsort()[-self.nMax:]

        self.idx2dMax[0,:] = self.pixelXVec[idxMaxSorted]
        self.idx2dMax[1,:] = self.pixelYVec[idxMaxSorted]    
            
        return self.idx2dMax


#------------------------------------------------------------------------------
def computeGradientMat(gradxMat, gradyMat, pixelOffsetMat, B, imgMat, maskMat, height, width):
    
    nP = pixelOffsetMat.shape[1]
    gradxMat *= 0
    gradyMat *= 0
    
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
    
    crossProdMat *= 0
    
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
    
    localMaxMat *= 0
    courseMaxVec *= 0
    pixelXVec *= 0
    pixelYVec *= 0
    
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
                    

#------------------------------------------------------------------------------
def rgbToGreyMat(rgbMat):
    return 0.299*rgbMat[:,:,0] + 0.587*rgbMat[:,:,1] + 0.114*rgbMat[:,:,2]
