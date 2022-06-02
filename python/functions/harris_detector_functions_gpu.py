import numpy as np
import cupy as cp
import cv_functions_gpu as cvGpu

"""----------------------------------------------------------------------------
                         ~~Cuda Kernel functions~~                          """
#------------------------------------------------------------------------------
# 5x5 gaussian blur
cp_blur = cp.RawKernel(r'''
extern "C" __global__
void gaussianBlur_5x5(float* imgBlurMat, // outputs
                      float* imgMat,     // inputs
                      float* maskMat, 
                      float* filt,
                      int height, 
                      int width) {
  
    int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    // check to make sure center pixel is valid, and there is range for the 3x3 window
    if ( (iCol >= 2)           &&
         (iCol < (width - 2))  &&
         (iRow >= 2)           &&
         (iRow < (height - 2)) &&
         (maskMat[iRow*width + iCol] > 0.)) {
    
        float maskSum = 
        filt[0]  * maskMat[(iRow-2)*width + iCol-2] + 
        filt[1]  * maskMat[(iRow-2)*width + iCol-1] + 
        filt[2]  * maskMat[(iRow-2)*width + iCol  ] + 
        filt[3]  * maskMat[(iRow-2)*width + iCol+1] + 
        filt[4]  * maskMat[(iRow-2)*width + iCol+2] + 
        filt[5]  * maskMat[(iRow-1)*width + iCol-2] + 
        filt[6]  * maskMat[(iRow-1)*width + iCol-1] + 
        filt[7]  * maskMat[(iRow-1)*width + iCol  ] + 
        filt[8]  * maskMat[(iRow-1)*width + iCol+1] + 
        filt[9]  * maskMat[(iRow-1)*width + iCol+2] + 
        filt[10] * maskMat[(iRow  )*width + iCol-2] + 
        filt[11] * maskMat[(iRow  )*width + iCol-1] + 
        filt[12] * maskMat[(iRow  )*width + iCol  ] + 
        filt[13] * maskMat[(iRow  )*width + iCol+1] + 
        filt[14] * maskMat[(iRow  )*width + iCol+2] + 
        filt[15] * maskMat[(iRow+1)*width + iCol-2] + 
        filt[16] * maskMat[(iRow+1)*width + iCol-1] + 
        filt[17] * maskMat[(iRow+1)*width + iCol  ] + 
        filt[18] * maskMat[(iRow+1)*width + iCol+1] + 
        filt[19] * maskMat[(iRow+1)*width + iCol+2] + 
        filt[20] * maskMat[(iRow+2)*width + iCol-2] + 
        filt[21] * maskMat[(iRow+2)*width + iCol-1] + 
        filt[22] * maskMat[(iRow+2)*width + iCol  ] + 
        filt[23] * maskMat[(iRow+2)*width + iCol+1] + 
        filt[24] * maskMat[(iRow+2)*width + iCol+2];
        
        float filtSum = 
        filt[0]  * maskMat[(iRow-2)*width + iCol-2] * imgMat[(iRow-2)*width + iCol-2] + 
        filt[1]  * maskMat[(iRow-2)*width + iCol-1] * imgMat[(iRow-2)*width + iCol-1] + 
        filt[2]  * maskMat[(iRow-2)*width + iCol  ] * imgMat[(iRow-2)*width + iCol  ] + 
        filt[3]  * maskMat[(iRow-2)*width + iCol+1] * imgMat[(iRow-2)*width + iCol+1] + 
        filt[4]  * maskMat[(iRow-2)*width + iCol+2] * imgMat[(iRow-2)*width + iCol+2] + 
        filt[5]  * maskMat[(iRow-1)*width + iCol-2] * imgMat[(iRow-1)*width + iCol-2] + 
        filt[6]  * maskMat[(iRow-1)*width + iCol-1] * imgMat[(iRow-1)*width + iCol-1] + 
        filt[7]  * maskMat[(iRow-1)*width + iCol  ] * imgMat[(iRow-1)*width + iCol  ] + 
        filt[8]  * maskMat[(iRow-1)*width + iCol+1] * imgMat[(iRow-1)*width + iCol+1] + 
        filt[9]  * maskMat[(iRow-1)*width + iCol+2] * imgMat[(iRow-1)*width + iCol+2] + 
        filt[10] * maskMat[(iRow  )*width + iCol-2] * imgMat[(iRow  )*width + iCol-2] + 
        filt[11] * maskMat[(iRow  )*width + iCol-1] * imgMat[(iRow  )*width + iCol-1] + 
        filt[12] * maskMat[(iRow  )*width + iCol  ] * imgMat[(iRow  )*width + iCol  ] + 
        filt[13] * maskMat[(iRow  )*width + iCol+1] * imgMat[(iRow  )*width + iCol+1] + 
        filt[14] * maskMat[(iRow  )*width + iCol+2] * imgMat[(iRow  )*width + iCol+2] + 
        filt[15] * maskMat[(iRow+1)*width + iCol-2] * imgMat[(iRow+1)*width + iCol-2] + 
        filt[16] * maskMat[(iRow+1)*width + iCol-1] * imgMat[(iRow+1)*width + iCol-1] + 
        filt[17] * maskMat[(iRow+1)*width + iCol  ] * imgMat[(iRow+1)*width + iCol  ] + 
        filt[18] * maskMat[(iRow+1)*width + iCol  ] * imgMat[(iRow+1)*width + iCol  ] + 
        filt[19] * maskMat[(iRow+1)*width + iCol+2] * imgMat[(iRow+1)*width + iCol+2] + 
        filt[20] * maskMat[(iRow+2)*width + iCol-2] * imgMat[(iRow+2)*width + iCol-2] + 
        filt[21] * maskMat[(iRow+2)*width + iCol-1] * imgMat[(iRow+2)*width + iCol-1] + 
        filt[22] * maskMat[(iRow+2)*width + iCol  ] * imgMat[(iRow+2)*width + iCol  ] + 
        filt[23] * maskMat[(iRow+2)*width + iCol+1] * imgMat[(iRow+2)*width + iCol+1] + 
        filt[24] * maskMat[(iRow+2)*width + iCol+2] * imgMat[(iRow+2)*width + iCol+2]; 

        if (maskSum > 0.){
            imgBlurMat[iRow*width + iCol] = filtSum / maskSum;
        }
    }
}
''', 'gaussianBlur_5x5')


#------------------------------------------------------------------------------
# gradient
cp_gradient = cp.RawKernel(r'''
extern "C" __global__
void computeGradient(float* gradxxMat,  // outputs
                     float* gradyyMat, 
                     float* gradxyMat, 
                     float* imgMat,     // inputs
                     float* maskMat, 
                     int height, 
                     int width) {
  
    int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    // check to make sure center pixel is valid, and there is range for the 3x3 window
    if ( (iCol >= 1)           &&
         (iCol < (width - 1))  &&
         (iRow >= 1)           &&
         (iRow < (height - 1)) &&
         (maskMat[iRow*width + iCol] == 1)) {
    
        // this implements a filter with potentially missing data, as specified by maskMat.  
        // disregard the entire row if either end pixel is missing
        // skip the center column because the coefficients are zero
        float rowMasks[3] = {
            (maskMat[(iRow-1)*width + iCol-1] * maskMat[(iRow-1)*width + iCol+1]), 
            (maskMat[(iRow  )*width + iCol-1] * maskMat[(iRow  )*width + iCol+1]), 
            (maskMat[(iRow+1)*width + iCol-1] * maskMat[(iRow+1)*width + iCol+1])  };

        // this is the convolution of the gx kernel over the 3x3 pixel window
        // skip the center column because the coefficients are zero
        float gradx = 
        1.0 * rowMasks[0] * imgMat[(iRow-1)*width + iCol-1] + 
       -1.0 * rowMasks[0] * imgMat[(iRow-1)*width + iCol+1] +
        2.0 * rowMasks[1] * imgMat[(iRow  )*width + iCol-1] + 
       -2.0 * rowMasks[1] * imgMat[(iRow  )*width + iCol+1] +
        1.0 * rowMasks[2] * imgMat[(iRow+1)*width + iCol-1] + 
       -1.0 * rowMasks[2] * imgMat[(iRow+1)*width + iCol+1];

        // disregard entire column if either end pixel is missing
        float colMasks[3] = {
            (maskMat[(iRow-1)*width + iCol-1] * maskMat[(iRow+1)*width + iCol-1]),  
            (maskMat[(iRow-1)*width + iCol  ] * maskMat[(iRow+1)*width + iCol  ]),  
            (maskMat[(iRow-1)*width + iCol+1] * maskMat[(iRow+1)*width + iCol+1])  };

        // this is the convolution of the gy kernel over the 3x3 pixel window
        // skip the center column because the coefficients are zero
        float grady = 
        1.0 * colMasks[0] * imgMat[(iRow-1)*width + iCol-1] + 
        2.0 * colMasks[1] * imgMat[(iRow-1)*width + iCol  ] + 
        1.0 * colMasks[2] * imgMat[(iRow-1)*width + iCol+1] +
       -1.0 * colMasks[0] * imgMat[(iRow+1)*width + iCol-1] + 
       -2.0 * colMasks[1] * imgMat[(iRow+1)*width + iCol  ] + 
       -1.0 * colMasks[2] * imgMat[(iRow+1)*width + iCol+1]; 
       
       gradxxMat[iRow*width + iCol] = gradx * gradx;
       gradyyMat[iRow*width + iCol] = grady * grady;
       gradxyMat[iRow*width + iCol] = gradx * grady;
   }
}
''', 'computeGradient')


#------------------------------------------------------------------------------
# 5x5 gaussian blur
cp_corner = cp.RawKernel(r'''
extern "C" __global__
void corner_repsonse(float* cornerMat,  // outputs
                      float* gradxxMat, // inputs
                      float* gradyyMat,
                      float* gradxyMat,
                      float* maskMat, 
                      int height, 
                      int width) {
  
    int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    // check to make sure center pixel is valid, and there is range for the 3x3 window
    if ( (iCol >= 0)       &&
         (iCol < (width))  &&
         (iRow >= 0)       &&
         (iRow < (height)) &&
         (maskMat[iRow*width + iCol] > 0.) ){
         
         float xx = gradxxMat[iRow*width + iCol];
         float yy = gradyyMat[iRow*width + iCol];
         float xy = gradxyMat[iRow*width + iCol];
         float trace = xx + yy;
         float det = xx*yy - xy*xy;
         cornerMat[iRow*width + iCol] =  det - 0.06 * trace * trace;
    }
}
''', 'corner_repsonse')


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
  
    if( (ii >= 2*c) && (ii < height-2*c) && (jj >= 2*c) && (jj < width-2*c) ) {
        
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

"""----------------------------------------------------------------------------
                              ~~Classes~~                                   """
#------------------------------------------------------------------------------
  
class HarrisDetectorGpu:

    def __init__(self, height, width, c, nMax): 
        
        # parameters
        self.height = height
        self.width = width
        self.c = c
        self.nMax = nMax

        # working variables
        self.gaussianBlurObj = cvGpu.gaussianBlur()
        self.blurMat   = cp.zeros((height, width), dtype = cp.float32)
        
        self.gradxxMat = cp.zeros((height, width), dtype = cp.float32)
        self.gradyyMat = cp.zeros((height, width), dtype = cp.float32)
        self.gradxyMat = cp.zeros((height, width), dtype = cp.float32)
        
        self.gradxxBlurMat = cp.zeros((height, width), dtype = cp.float32)
        self.gradyyBlurMat = cp.zeros((height, width), dtype = cp.float32)
        self.gradxyBlurMat = cp.zeros((height, width), dtype = cp.float32)

        self.cornerMat = cp.zeros((height, width), dtype = cp.float32)
        
        self.cornerMaxMat = cp.zeros((height, width), dtype = cp.float32)
        height_c = int(height / self.c)
        width_c = int(width / self.c)
        self.courseMaxVec = cp.zeros(height_c * width_c, dtype = cp.float32)
        self.pixelXVec = cp.zeros(height_c * width_c, dtype = cp.int32)
        self.pixelYVec = cp.zeros(height_c * width_c, dtype = cp.int32)
        
    def findCornerPoints(self, cornerPointIdx, imgMat, maskMat):
        
        grad(self.gradxxMat, \
             self.gradyyMat, \
             self.gradxyMat, \
             imgMat, \
             maskMat, \
             self.height, \
             self.width)            

        self.gaussianBlurObj.blurImg(self.gradxxBlurMat, self.gradxxMat, maskMat)
        self.gaussianBlurObj.blurImg(self.gradyyBlurMat, self.gradyyMat, maskMat)
        self.gaussianBlurObj.blurImg(self.gradxyBlurMat, self.gradxyMat, maskMat)
                
        corner(self.cornerMat, \
               self.gradxxBlurMat, \
               self.gradyyBlurMat, \
               self.gradxyBlurMat, \
               maskMat, \
               self.height, \
               self.width)
            
        findLocalMax(self.cornerMaxMat,  \
                    self.courseMaxVec,  \
                    self.pixelXVec,     \
                    self.pixelYVec,     \
                    self.cornerMat,  \
                    self.c,      \
                    self.height, \
                    self.width)
            
        self.courseMaxVec_cpu = self.courseMaxVec.get()
        idxMaxSorted = self.courseMaxVec_cpu.argsort()[-self.nMax:]

        cornerPointIdx[0,:] = self.pixelXVec[idxMaxSorted]
        cornerPointIdx[1,:] = self.pixelYVec[idxMaxSorted]


"""----------------------------------------------------------------------------
                             ~~functions~~                                  """
#------------------------------------------------------------------------------
def blur(imgBlurMat, imgMat, maskMat, filt, height, width):
    
    imgBlurMat *= 0
    
    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_blur(grid, block,  \
            (imgBlurMat,        # outputs
             imgMat,            # inputs
             maskMat,
             filt,
             height, 
             width ))
                
        
def grad(gradxxMat, gradyyMat, gradxyMat, imgMat, maskMat, height, width):
    
    gradxxMat *= 0
    gradyyMat *= 0
    gradxyMat *= 0
    
    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_gradient(grid, block,  \
               (gradxxMat,          # outputs
                gradyyMat,
                gradxyMat,
                imgMat,            # inputs
                maskMat,
                height, 
                width ))
        
def corner(cornerMat, gradxxMat, gradyyMat, gradxyMat, maskMat, height, width):
    
    cornerMat *= 0
    
    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_corner(grid, block,  \
               (cornerMat,   # outputs
                gradxxMat,   # inputs      
                gradyyMat,
                gradxyMat,           
                maskMat,
                height, 
                width ))        

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
      
def rgb_to_grey(rgbMat):
    return 0.299*rgbMat[:,:,0] + 0.587*rgbMat[:,:,1] + 0.114*rgbMat[:,:,2]
