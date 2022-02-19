import matplotlib.pyplot as plt
import numpy as np
import time

###############################################################################
# Functions 

#------------------------------------------------------------------------------
def quadraticSurface(f, B):
    
    fScaled = (f - f[-1])
    
    # compute least squares coefficients for quadratic cost surface
    K = np.matmul(B, fScaled[0:-1])
    
    # gradient
    gradient = K[[0,1]]
    
    # compute eigen values of the surface
    a = K[3]
    c = K[2]/2
    b = K[4]
    
    D = np.sqrt((a-b)**2 + 4*c*c)
    eig1 = 0.5 * (a + b + D)  # this should be the maximum eigvec
    eig2 = 0.5 * (a + b - D)
    
    eigVec1 = np.array([1, b - a + D])
    eigVec1  = np.abs(eigVec1) # put the eigen vector in the the first quadrant
    eigVec1 = eig1 * eigVec1 / np.sqrt(np.sum(eigVec1 * eigVec1))
    
    return eig1, eig2, eigVec1, gradient


#------------------------------------------------------------------------------
def findCornerPoints(greyMat, maskMat, pixelOffsetMat, B, nMax):
    
    height, width = greyMat.shape
    
    gradxMat = np.zeros((height, width))
    gradyMat = np.zeros((height, width))
    gradMat = np.zeros((height, width))
    crossProdxMat = np.zeros((height, width))
    crossProdyMat = np.zeros((height, width))
    
    f = np.zeros((nP))
    maskVec = np.zeros((nP))
    
    # eigen surface for each pixel
    for ii in range(3, height-3):
        for jj in range(3, width-3):
            
            for p in range(nP):
                
                x = ii + pixelOffsetMat[p,0]
                y = jj + pixelOffsetMat[p,1]
                
                f[p] = greyMat[x, y]
                maskVec[p] = maskMat[x, y]
              
            if maskVec.sum() == nP:
                eig1, eig2, eigVec1, gradient = quadraticSurface(f, B)
                
                gradxMat[ii,jj] = gradient[0] 
                gradyMat[ii,jj] = gradient[1] 
    
    # change in gradient for each pixel
    # eigen surface for each pixel
    l = 2
    for ii in range(3, height-3):
        for jj in range(3, width-3):
            
            gradx = 1 * (gradxMat[ii-l,jj-l] * gradyMat[ii+l,jj-l] - gradyMat[ii-l,jj-l] * gradxMat[ii+l,jj-l]) + \
                    2 * (gradxMat[ii-l,jj  ] * gradyMat[ii+l,jj  ] - gradyMat[ii-l,jj  ] * gradxMat[ii+l,jj  ]) + \
                    1 * (gradxMat[ii-l,jj+l] * gradyMat[ii+l,jj+l] - gradyMat[ii-l,jj-l] * gradxMat[ii+l,jj-l]) 
            
            
            grady = 1 * (gradxMat[ii-l,jj-l] * gradyMat[ii-l,jj+l] - gradyMat[ii-l,jj-l] * gradxMat[ii-l,jj+l]) + \
                    2 * (gradxMat[ii,  jj-l] * gradyMat[ii,  jj+l] - gradyMat[ii,  jj-l] * gradxMat[ii,  jj+l]) + \
                    1 * (gradxMat[ii+l,jj-l] * gradyMat[ii+l,jj+l] - gradyMat[ii+l,jj-l] * gradxMat[ii+l,jj+l])
            
            crossProdxMat[ii,jj]  = gradx 
            crossProdyMat[ii,jj] = grady
            gradMat[ii,jj] = np.sqrt(gradx*gradx + grady*grady)
            
    
    # coarse grid
    c = 8
    thresh = 130
    coarseGradMat = np.zeros((height, width))
    #cornerMask = np.full((height, width), False, dtype = bool)
    winMask = np.full((height, width), False, dtype = bool)
    for ii in range(c, height-c):
        for jj in range(c, width-c):
    
            if gradMat[ii, jj] > thresh:
                
                subMat = gradMat[ii-c:ii+c, jj-c:jj+c]
                
                if gradMat[ii,jj] >= subMat.max():
                    coarseGradMat[ii, jj] = gradMat[ii, jj]
                    #cornerMask[ii, jj] = True
                    #winMask[ii-c:ii+c, jj-c:jj+c] = True
                    
    vec = coarseGradMat.flatten()
    
    start = time.time()
    vecArgSort = vec.argsort()
    print('sort timer:', time.time() - start)
       
    idxMax = vecArgSort[-nMax:]  # argsort puts the maximum value at the end
    idx2dMax = np.array(np.unravel_index(idxMax, (height, width)))

    vec2 = np.full((height * width), False, dtype = bool)
    vec2[idxMax] = True
    cornerMask = vec2.reshape((height, width))
    
    for ii in range(c, height-c):
        for jj in range(c, width-c):
            if cornerMask[ii, jj]:
                
                winMask[ii-c:ii+c, jj-c:jj+c] = True
                
    
                    
    return cornerMask, winMask, coarseGradMat, idx2dMax, gradMat


#--------------------------------------------------------------------------
def computeMatError(mat1, mat2, mask):
    matErr = mat1 - mat2
    f = np.sum(matErr[mask] * matErr[mask]) / np.sum(mask).astype(float)
    #f = np.sum(np.abs(matErr))
    return f


#------------------------------------------------------------------------------
def computeMatchCost(rgb1Mat, rgb2Mat, mask1Mat, mask2Mat, id2dMax1, id2dMax2):
    
    l = 8

    mask = np.logical_and(mask1Mat[id2dMax1[0]-l:id2dMax1[0]+l, id2dMax1[1]-l:id2dMax1[1]+l], mask2Mat[id2dMax2[0]-l:id2dMax2[0]+l, id2dMax2[1]-l:id2dMax2[1]+l])
    
    fr = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l, id2dMax1[1]-l:id2dMax1[1]+l, 0], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l, id2dMax2[1]-l:id2dMax2[1]+l, 0], mask)
    fg = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l, id2dMax1[1]-l:id2dMax1[1]+l, 0], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l, id2dMax2[1]-l:id2dMax2[1]+l, 0], mask)
    fb = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l, id2dMax1[1]-l:id2dMax1[1]+l, 0], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l, id2dMax2[1]-l:id2dMax2[1]+l, 0], mask)
    
    f = fr + fg + fb
    return f


#--------------------------------------------------------------------------
def drawBox(rgbMat, x, y, l):

    c = np.array([200, 0, 200], dtype = np.ubyte)
    rgbMat[x-l:x+l, y-l, :] = c
    rgbMat[x-l:x+l, y+l, :] = c
    
    rgbMat[x-l, y-l:y+l, :] = c
    rgbMat[x+l, y-l:y+l, :] = c
    
 
#------------------------------------------------------------------------------
###############################################################################
# Testing

nMax = 100

# the FAST detector offset, with the 0,0 point added at the end
pixelOffsetMat = np.array([[-3,  0],
                        [-3,  1],
                        [-2,  2],
                        [-1,  3],
                        [ 0,  3],
                        [ 1,  3],
                        [ 2,  2],
                        [ 3,  1],
                        [ 3,  0],
                        [ 3, -1],
                        [ 2, -2],
                        [ 1, -3],
                        [ 0, -3],
                        [-1, -3],
                        [-2, -2],
                        [-3, -1],
                        [ 0,  0]])

# create least squares matrix for computing quadratic model of cost function from offset error pointsn
nP, _ = pixelOffsetMat.shape 
A = np.zeros((nP-1, 5))
for offset in range(nP-1):
    x = pixelOffsetMat[offset, 0]
    y = pixelOffsetMat[offset, 1]
    
    A[offset, :] = np.array([x, y, x*y, x*x, y*y])

B = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
    
corner1Mask, win1Mask, coarseGradMat, idx2dMax1, gradMat = findCornerPoints(greyMat, maskMat, pixelOffsetMat, B, nMax)
rgb1Win = np.copy(rgbBlurMat)
rgb1Win[np.invert(win1Mask)] = 0  

corner2Mask, win2Mask, coarseGradMat, idx2dMax2, grad2Mat = findCornerPoints(grey2Mat, mask2Mat, pixelOffsetMat, B, nMax)
rgb2Win = np.copy(rgb2BlurMat)
rgb2Win[np.invert(win2Mask)] = 0  

costMatrix = np.zeros((nMax, nMax))
for id1 in range(nMax):
    for id2 in range(nMax):
        
        costMatrix[id1, id2] = computeMatchCost(rgbBlurMat, rgb2BlurMat, maskMat, mask2Mat, idx2dMax1[:,id1], idx2dMax2[:,id2])

colMin = costMatrix.argmin(axis = 0)
fMin = costMatrix.min(axis = 0)
               
# visualize matches
matchId = 92

im1x = idx2dMax1[0, colMin[matchId]]
im1y = idx2dMax1[1, colMin[matchId]]

im2x = idx2dMax2[0, matchId]
im2y = idx2dMax2[1, matchId]

rgb1Match = np.copy(rgbBlurMat)
drawBox(rgb1Match, im1x, im1y, 9)

rgb2Match = np.copy(rgb2BlurMat)
drawBox(rgb2Match, im2x, im2y, 9)

plt.figure('rgb  match' + str(frame1))
plt.title('rgb match ' + str(frame1))
plt.imshow(rgb1Match)


plt.figure('rgb  match' + str(frame2))
plt.title('rgb match ' + str(frame2))
plt.imshow(rgb2Match)

        
#m1Mat = np.clip(m1Mat, -100, 100)
gradNormMat = cvFun.normalizeEntropymat(gradMat, Hmask, 0, 10)


plt.figure('rgb  win frame' + str(frame1))
plt.title('rgb win frame ' + str(frame1))
plt.imshow(rgb1Win)

plt.figure('rgb  win frame' + str(frame2))
plt.title('rgb win frame ' + str(frame2))
plt.imshow(rgb2Win)

plt.figure('gradNormMat')
plt.title('gradNormMat')
plt.imshow(gradNormMat)

plt.figure('costMatrix')
plt.title('costMatrix')
plt.hist(costMatrix.flatten(), density=True, bins = 300)

    
plt.figure('coarseGradMat hist')
plt.title('coarseGradMat hist')
plt.hist(gradMat.flatten(), density=True, bins = 300)