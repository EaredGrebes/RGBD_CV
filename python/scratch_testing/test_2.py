import cv_functions as cvFun
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------
# create offset vector, last point is the center point, this is important
def courseGrid():
    
    radius = 15
    resVec = np.linspace(-radius, radius, 10)
    offsetMat = []
    
    for ii in resVec:
        for jj in resVec:
            if np.sqrt(ii*ii + jj*jj) < radius:
                offsetMat.append([ii, jj])

    offsetMat = np.array(offsetMat, dtype = int)
    
    return offsetMat


def circleGrid():
    c = np.sqrt(2)
    rad = 10
    offsetMat = np.array([[-1/c, -1/c],
                           [-1,  0],
                           [-1/c,  1/c],
                           [ 0, -1],
                           [ 0,  1],
                           [ 1/c, -1/c],
                           [ 1,  0],
                           [ 1/c,  1/c],
                           [ 0,  0]]) * rad
    
    offsetMat = np.array(offsetMat, dtype = int)
    return offsetMat


# finite difference of frame1 wrt frame2 of each coarse grid block
hc, wc = coarseGridHMaskMat.shape

# the coarse grid pixel location
iic = 9
jjc = 18

########################
# Coarse Measurement

offsetMat = courseGrid()
nP, _ = offsetMat.shape  # number of finite difference points 

# for now, set location of the second image search point to the first image
xPixel = iic * l 
yPixel = jjc * l 

index2 = [xPixel, yPixel]
fVec = np.zeros(nP)
for p in range(nP):
    
    index1 = [iic*l, jjc*l]
    
    offset = offsetMat[p,:]
    fVec[p] = cvFun.computeRgbMatShiftError(rgbBlurMat,     \
                                            rgb2BlurMat,    \
                                            xyzMat[:,:,2],  \
                                            xyz2Mat[:,:,2], \
                                            maskMat,        \
                                            mask2Mat,       \
                                            l,              \
                                            index1, \
                                            index2, \
                                            offset)
        
pixelOffset = offsetMat[np.argmin(fVec), :]
print(pixelOffset)


plt.figure()
plt.plot(offsetMat[:,0], offsetMat[:,1], '.')

plt.close('all')

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(offsetMat[:,0], offsetMat[:,1], fVec, '.', color = 'm')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.figure('Coarse Grid Mask')
plt.title('Coarse Grid Mask')
plt.spy(coarseGridHMaskMat)