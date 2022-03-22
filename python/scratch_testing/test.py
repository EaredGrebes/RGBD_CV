import cv_functions as cvFun

# finite difference of frame1 wrt frame2 of each coarse grid block
hc, wc = coarseGridHMaskMat.shape
buffer = 1 # number of coarse edge pixels to ignore

# create offset vector, last point is the center point, this is important
offsetSchedule = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
maxTravel = 3

c = np.sqrt(2)
offsetMat0 = np.array([[-1/c, -1/c],
                       [-1,  0],
                       [-1/c,  1/c],
                       [ 0, -1],
                       [ 0,  1],
                       [ 1/c, -1/c],
                       [ 1,  0],
                       [ 1/c,  1/c],
                       [ 0,  0]])

nP, _ = offsetMat0.shape  # number of finite difference points  

# this is the coarse pixel match between the first rgb image and the second rgb image
xDeltaPixel = np.zeros((hc, wc))
yDeltaPixel = np.zeros((hc, wc))

xPixelMatch = np.zeros((hc, wc), dtype = int)
yPixelMatch = np.zeros((hc, wc), dtype = int)

xDeltaBlurPixel = np.zeros((hc, wc), dtype = int)
yDeltaBlurPixel = np.zeros((hc, wc), dtype = int)

blockRangeHeight = np.arange(buffer, hc-buffer)
blockRangeWidth = np.arange(buffer, wc-buffer)

#blockRangeHeight = np.array([iic])
#blockRangeWidth =  np.array([jjc])

# initialize the pixel map
for iic in blockRangeHeight:
    for jjc in blockRangeWidth: 
        if coarseGridHMaskMat[iic, jjc]:
        
            xPixelMatch[iic, jjc] = iic*l
            yPixelMatch[iic, jjc] = jjc*l
            
print('yNew', yPixelMatch[iic, jjc])    
        
# the big optimizer loop
start = time.time()
for optIter in range(len(offsetSchedule)): 
    
    offsetDist = offsetSchedule[optIter]

    # errorTensor stores the error for each offset point
    errorTensor = np.zeros((hc, wc, nP))   
    
    offsetMat = np.round(offsetMat0 * offsetDist).astype(int)
    
    # create least squares matrix for computing quadratic model of cost function from offset error points  
    A = np.zeros((nP-1, 5))
    for offset in range(nP-1):
        x = offsetMat[offset, 0]
        y = offsetMat[offset, 1]
        
        A[offset, :] = np.array([x, y, x*y, x*x, y*y])

    B = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
    
    for iic in blockRangeHeight:
        for jjc in blockRangeWidth: 
            if coarseGridHMaskMat[iic, jjc]:
                
                xPixel = xPixelMatch[iic, jjc] + xDeltaBlurPixel[iic, jjc]
                yPixel = yPixelMatch[iic, jjc] + yDeltaBlurPixel[iic, jjc]
                
                if (xPixel - offsetDist) > 0 and (xPixel + offsetDist + l) < height-1 and (yPixel - offsetDist) > 0 and (yPixel + offsetDist + l) < width-1:
                    for p in range(nP):
                        
                        index1 = [iic*l, jjc*l]
                        index2 = [xPixel, yPixel]
                        offset = offsetMat[p,:]
                        errorTensor[iic, jjc, p] = cvFun.computeRgbMatShiftError(rgbBlurMat,     \
                                                                                 rgb2BlurMat,    \
                                                                                 xyzMat[:,:,2],  \
                                                                                 xyz2Mat[:,:,2], \
                                                                                 maskMat,        \
                                                                                 mask2Mat,       \
                                                                                 l,              \
                                                                                 index1, \
                                                                                 index2, \
                                                                                 offset)
         
    for iic in blockRangeHeight:
        for jjc in blockRangeWidth: 
            if coarseGridHMaskMat[iic, jjc]:
                
                f = errorTensor[iic,jjc,:]
                deltaPixels = cvFun.computeDeltaPixelStep(f, B, offsetMat, offsetDist)  
                
                xDeltaPixel[iic, jjc] += deltaPixels[0]
                yDeltaPixel[iic, jjc] += deltaPixels[1]
                
    xDeltaBlurPixel = myCv.blurMat(xDeltaPixel, coarseGridHMaskMat).astype(int)
    yDeltaBlurPixel = myCv.blurMat(yDeltaPixel, coarseGridHMaskMat).astype(int)
                
xPixelMatch = xPixelMatch + xDeltaBlurPixel                
yPixelMatch = yPixelMatch + yDeltaBlurPixel
                            
print('timer:', time.time() - start)
    
iic = 16
jjc = 18

f = errorTensor[iic,jjc,:]

fScaled = (f - f[-1])
x = offsetMat[:,0]
y = offsetMat[:,1]

K = np.matmul(B, fScaled[0:-1])

# create 2-D quadratic surface
vec = np.linspace(-offsetDist, offsetDist, 100)
xx, yy = np.meshgrid(vec, vec)
fQuad = K[0]*xx + K[1]*yy + K[2]*xx*yy + K[3]*xx*xx + K[4]*yy*yy

xDeltaPixel = xDeltaBlurPixel - xDeltaBlurPixel.min()
yDeltaPixel = yDeltaBlurPixel - yDeltaBlurPixel.min()

plt.close('all')

# the first image
ir1 = iic*l
ic1 = jjc*l

# the image we are trying to find a match to
ir2 = xPixelMatch[iic, jjc] 
ic2 = yPixelMatch[iic, jjc]

plt.figure()
plt.imshow(rgbBlurMat[ir1:ir1+l, ic1:ic1+l, :])
    
plt.figure()
plt.imshow(rgb2BlurMat[ir2:ir2+l, ic2:ic2+l, :])

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, fScaled, color = 'm')
ax.plot_surface(xx, yy, fQuad)
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.figure('Coarse Grid Mask')
plt.title('Coarse Grid Mask')
plt.spy(coarseGridHMaskMat)

plt.figure('X delta pixel')
plt.title('X delta pixel')
plt.imshow(xDeltaBlurPixel)

plt.figure('Y delta pixel')
plt.title('Y delta pixel')
plt.imshow(yDeltaPixel)