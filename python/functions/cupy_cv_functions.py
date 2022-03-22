import numpy as np
import cupy as cp


cp_add_matricies = cp.RawKernel(r'''
extern "C" __global__
void add_arrs(float* x1, float* x2, float* y, int width, int height){
  
  int iCol = blockDim.x * blockIdx.x + threadIdx.x;
  int iRow = blockDim.y * blockIdx.y + threadIdx.y;
  
  if(iCol < width && iRow < height){
    y[iRow + iCol*height] = x1[iRow + iCol*height] + x2[iRow + iCol*height];
  }
}
''', 'add_arrs')

width = 848
height = 480

x1_cpu = np.random.rand(height, width)
x2_cpu = np.random.rand(height, width)

# move the data to the current device.
x1_gpu = cp.array(x1_cpu, dtype = cp.float32, copy=False)  
x2_gpu = cp.array(x2_cpu, dtype = cp.float32, copy=False)  

# initialize output
y = cp.zeros((height, width), dtype=cp.float32)

# Grid and block sizes
block = (8, 8)
grid = (int(width/block[0]), int(height/block[1]))

# Call kernel
cp_add_matricies(grid, block, ( x1_gpu, x2_gpu, y, cp.int32(width), cp.int32(height) ) )

# review results
y_cpu = y.get()
y_check = x1_cpu + x2_cpu
check = np.isclose(y_cpu, y_check)


