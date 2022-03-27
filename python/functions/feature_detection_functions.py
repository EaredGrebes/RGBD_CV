import numpy as np

#------------------------------------------------------------------------------
  
class corner_detector_class:

    def __init__(self): 
        
        # the FAST detector offset, with the 0,0 point added at the end   
        self.pixelOffsetMat = np.array([[-3, -3, -2, -1,  0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3, 0], \
                                        [ 0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3, -3, -3, -2, -1, 0]])            

        # create least squares matrix for computing quadratic model of cost function from offset error pointsn
        nP = self.pixelOffsetMat.shape[1]
        A = np.zeros((nP-1, 5))
        for offset in range(nP-1):
            x = self.pixelOffsetMat[0, offset]
            y = self.pixelOffsetMat[1, offset]
            
            A[offset, :] = np.array([x, y, x*y, x*x, y*y])

        self.B = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        
        
    def findCornerPoints(self, greyMat, maskMat, nMax):
        return findCornerPoints(greyMat, maskMat, self.pixelOffsetMat, self.B, nMax)



#------------------------------------------------------------------------------
def quadraticSurface(f, B):
    
    # [0, 0] point added at the end
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
def computeGradientMat(pixelOffsetMat, B, imgMat, maskMat, height, width):
    
    nP = pixelOffsetMat.shape[1]
    
    gradxMat = np.zeros((height, width))
    gradyMat = np.zeros((height, width))
    f = np.zeros((nP))
    maskVec = np.zeros((nP))
    
    # eigen surface for each pixel
    buf = 3
    for ii in range(buf, height-buf):
        for jj in range(buf, width-buf):
            
            for p in range(nP):
                
                x = ii + pixelOffsetMat[0,p]
                y = jj + pixelOffsetMat[1,p]
                
                f[p] = imgMat[x, y]
                maskVec[p] = maskMat[x, y]
              
            if maskVec.sum() == nP:
                eig1, eig2, eigVec1, gradient = quadraticSurface(f, B)
                
                gradxMat[ii,jj] = gradient[0] 
                gradyMat[ii,jj] = gradient[1] 
                
    return gradxMat, gradyMat

#------------------------------------------------------------------------------
def computeCrossProdMat(gradxMat, gradyMat, height, width):

    crossProdMat = np.zeros((height, width))

    # change in gradient for each pixel
    # eigen surface for each pixel
    l = 2
    buf = 3
    for ii in range(buf, height-buf):
        for jj in range(buf, width-buf):
            
            crossProdx = 1 * (gradxMat[ii-l,jj-l] * gradyMat[ii+l,jj-l] - gradyMat[ii-l,jj-l] * gradxMat[ii+l,jj-l]) + \
                         2 * (gradxMat[ii-l,jj  ] * gradyMat[ii+l,jj  ] - gradyMat[ii-l,jj  ] * gradxMat[ii+l,jj  ]) + \
                         1 * (gradxMat[ii-l,jj+l] * gradyMat[ii+l,jj+l] - gradyMat[ii-l,jj-l] * gradxMat[ii+l,jj-l]) 
            
            
            crossPrody = 1 * (gradxMat[ii-l,jj-l] * gradyMat[ii-l,jj+l] - gradyMat[ii-l,jj-l] * gradxMat[ii-l,jj+l]) + \
                         2 * (gradxMat[ii,  jj-l] * gradyMat[ii,  jj+l] - gradyMat[ii,  jj-l] * gradxMat[ii,  jj+l]) + \
                         1 * (gradxMat[ii+l,jj-l] * gradyMat[ii+l,jj+l] - gradyMat[ii+l,jj-l] * gradxMat[ii+l,jj+l])
            
            crossProdMat[ii,jj] = np.sqrt(crossProdx*crossProdx + crossPrody*crossPrody)
            
    return crossProdMat
 
#------------------------------------------------------------------------------
def findLocalMax(mat, c, height, width):
    
    # local maximum / course maximum
    coarseMaxMat = np.zeros((height, width))

    for ii in range(c, height-c-1):
        for jj in range(c, width-c-1):

            subMat = mat[ii-c:ii+c+1, jj-c:jj+c+1]
            
            if mat[ii,jj] >= subMat.max():
                coarseMaxMat[ii, jj] = mat[ii, jj]
                
    return coarseMaxMat      
    
    
#------------------------------------------------------------------------------
def findCornerPoints(greyMat, maskMat, pixelOffsetMat, B, nMax):
    
    c = 8 # local scale
    height, width = greyMat.shape
    
    gradxMat, gradyMat = computeGradientMat(pixelOffsetMat, B, greyMat, maskMat, height, width)
    
    crossProdMat = computeCrossProdMat(gradxMat, gradyMat, height, width)
            
    coarseGradMat = findLocalMax(crossProdMat, c, height, width)
        
    vec = coarseGradMat.flatten()
    vecArgSort = vec.argsort()

    idxMax = vecArgSort[-nMax:]  # argsort puts the maximum value at the end
    idx2dMax = np.array(np.unravel_index(idxMax, (height, width)))

    vec2 = np.full((height * width), False, dtype = bool)
    vec2[idxMax] = True
    cornerMask = vec2.reshape((height, width))
                    
    return cornerMask, coarseGradMat, idx2dMax, crossProdMat


#--------------------------------------------------------------------------
def computeMatError(mat1, mat2, mask):
    matErr = mat1 - mat2
    f = np.sum(matErr[mask] * matErr[mask]) / np.sum(mask).astype(float)
    #f = np.sum(np.abs(matErr))
    return f


#------------------------------------------------------------------------------
def computeMatchCost(rgb1Mat, rgb2Mat, mask1Mat, mask2Mat, id2dMax1, id2dMax2):
    
    l = 5
    maskMat1 = mask1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1]
    maskMat2 = mask2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1]
    mask = np.logical_and(maskMat1, maskMat2)
    
    fr = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1, 0], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1, 0], mask)
    fg = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1, 1], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1, 1], mask)
    fb = computeMatError(rgb1Mat[id2dMax1[0]-l:id2dMax1[0]+l+1, id2dMax1[1]-l:id2dMax1[1]+l+1, 2], rgb2Mat[id2dMax2[0]-l:id2dMax2[0]+l+1, id2dMax2[1]-l:id2dMax2[1]+l+1, 2], mask)
    
    f = fr + fg + fb
    return f


#--------------------------------------------------------------------------
def drawBox(rgbMat, x, y, l, c):

    rgbMat[x-l:x+l, y-l, :] = c
    rgbMat[x-l:x+l, y+l, :] = c
    
    rgbMat[x-l, y-l:y+l, :] = c
    rgbMat[x+l, y-l:y+l, :] = c