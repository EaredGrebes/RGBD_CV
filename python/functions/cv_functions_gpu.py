import numpy as np
import cupy as cp

"""----------------------------------------------------------------------------
                         ~~Cuda Kernel functions~~                          """
#------------------------------------------------------------------------------
# 5x5 gaussian blur
cp_convolve5x5 = cp.RawKernel(r'''
extern "C" __global__
void convolve5x5(float* imgBlurMat, // outputs
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
''', 'convolve5x5')


"""----------------------------------------------------------------------------
                    ~~kernel wrapper functions~~                            """
#------------------------------------------------------------------------------
def convolve5x5_kernelWrapper(imgBlurMat, imgMat, maskMat, filt, height, width):
    
    imgBlurMat *= 0
    
    # Grid and block sizes
    block = (8, 8)
    grid = (int(width/block[0]), int(height/block[1]))

    # Call kernel
    cp_convolve5x5(grid, block,  \
            (imgBlurMat,        # outputs
             imgMat,            # inputs
             maskMat,
             filt,
             height, 
             width ))
        
"""----------------------------------------------------------------------------
                     ~~API functions functions~~                            """
#------------------------------------------------------------------------------     
class gaussianBlur:

    def __init__(self): 
        self.gK = cp.array([[1, 4,  7,  4,  1],
                            [4, 16, 26, 16, 4],
                            [7, 26, 41, 26, 7],
                            [4, 16, 26, 16, 4],
                            [1, 4,  7,  4,  1]], dtype = cp.float32)
        self.filt = cp.array( self.gK.flatten() / np.sum(self.gK) )
        
    def blurImg(self, imgBlurMat, imgMat, maskMat):
        height, width = imgMat.shape
        convolve5x5_kernelWrapper(imgBlurMat, imgMat, maskMat, self.filt, height, width)
        
    def return_blurImg(self, imgMat, maskMat):
        height, width = imgMat.shape
        imgBlurMat = cp.zeros((height, width), dtype = cp.float32)
        convolve5x5_kernelWrapper(imgBlurMat, imgMat, maskMat, self.filt, height, width)      
        return imgBlurMat
        
        
   