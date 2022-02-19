import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cv_functions as cvFun
import time

#------------------------------------------------------------------------------
# cuda kernels
cp_add_matricies = cp.RawKernel(r'''
extern "C" __global__
void add_mats(float* x1, float* x2, float* y, int width, int height){
  
  int iCol = blockDim.x * blockIdx.x + threadIdx.x;
  int iRow = blockDim.y * blockIdx.y + threadIdx.y;
  
  if(iCol < width && iRow < height){
    y[iRow*width + iCol] = x1[iRow*width + iCol] + x2[iRow*width + iCol];
  }
}
''', 'add_mats')


cp_filter_image = cp.RawKernel(r'''
extern "C" __global__
void filter_image(float* imOut, float* imIn, bool* mask, float* filter, int imWidth, int imHeight, int filtWidth, int filtHeight){
  
    int iCol = blockIdx.x * blockDim.x  + threadIdx.x;
    int iRow = blockIdx.y * blockDim.y  + threadIdx.y;
  
    if( (iCol >= filtWidth)              &&
        (iCol <  (imWidth - filtWidth))  && 
        (iRow >= filtHeight)             &&
        (iRow <  (imHeight - filtHeight)) ){

    if (mask[iRow*imWidth + iCol]) {
            
        float sum = 0;
        float filtSum = 0;
        int filtW = filtWidth * 2 + 1;
        
        int iIm, jIm, iF, jF;
        
        for (int ii = -filtWidth; ii <= filtWidth; ii++) {
            for (int jj = -filtHeight; jj <= filtHeight; jj++) {
               
                iIm = iRow + ii;
                jIm = iCol + jj;
                
                iF = ii + filtWidth;
                jF = jj + filtHeight;
                
                if (mask[iIm*imWidth + jIm]) {
                    sum = sum + imIn[iIm*imWidth + jIm] * filter[iF*filtW + jF];  
                    filtSum = filtSum + filter[iF*filtW + jF];
                }
            }
        }
        
        if (filtSum > 0){
            imOut[iRow*imWidth + iCol] = sum / filtSum;
        }
    }
  }
}
''', 'filter_image')


width = 848
height = 480

filtWidth = 2
filtHeight = 2

imIn_cpu = np.random.rand(height, width)
maskMat = np.full((height, width), True)

maskMat[300:350, 400:450] = False

filterGaussian = np.array([ [1, 4,  7,  4,  1],
                            [4, 16, 26, 16, 4],
                            [7, 26, 41, 26, 7],
                            [4, 16, 26, 16, 4],
                            [1, 4,  7,  4,  1]]);

# move the data to the current device.
imIn_gpu = cp.array(imIn_cpu, dtype = cp.float32, copy=False)  
mask_gpu = cp.array(maskMat, dtype = cp.bool_, copy=False)  
filt_gpu = cp.array(filterGaussian, dtype = cp.float32, copy=False)  

# initialize output
imOut_gpu = cp.zeros((height, width), dtype=cp.float32)

# Grid and block sizes
block = (8, 8)
grid = (int(width/block[0]), int(height/block[1]))

# Call kernel
start = time.time()

cp_filter_image(grid, block,  \
(   imOut_gpu,           # outputs
    imIn_gpu,            # inputs
    mask_gpu,
    filt_gpu, 
    cp.int32(width), 
    cp.int32(height),
    cp.int32(filtWidth), 
    cp.int32(filtHeight) ) )
    
imOut_cpu = imOut_gpu.get()  
print('timer:', time.time() - start)  
    
# compare with other methods
myCv = cvFun.myCv(height, width) 

start = time.time()
blurMat = myCv.blurMat(imIn_cpu, maskMat)
print('timer:', time.time() - start) 

percentErr = np.abs(blurMat - imOut_cpu) / imIn_cpu

# review results
plt.close('all')

plt.figure()
plt.title('Image in')
plt.imshow(imIn_cpu)

plt.figure()
plt.title('GPU')
plt.imshow(imOut_cpu)

plt.figure()
plt.title('CPU')
plt.imshow(blurMat)

