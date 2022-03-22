import numpy as np
import matplotlib.pyplot as plt

# ~~ helper functions ~~    
#------------------------------------------------------------------------------
# here dim specifies the square side length of the kernel L as:
    # L = 2*dim + 1
def conv(K, Mat, iRow, iCol, dim):
    
    gKm = K * Mat[iRow-dim:iRow+dim+1, iCol-dim:iCol+dim+1]
    sumK = np.sum(gKm)
    
    return gKm, sumK


#--------------------------------------------------------------------------
def det3x3Symetric(A):
    
    a = A[0,0]
    b = A[0,1]
    c = A[0,2]
    d = A[1,1]
    e = A[1,2]
    f = A[2,2]
    
    return a * (d*f - e*e) + b * (c*e - b*f) + c * (b*e - d * c)


#--------------------------------------------------------------------------
def computeMatError(mat1, mat2, mask):
    matErr = mat1 - mat2
    f = np.sum(matErr[mask] * matErr[mask]) / np.sum(mask).astype(float)
    #f = np.sum(np.abs(matErr))
    return f


#--------------------------------------------------------------------------
def computeDeltaPixelStep(f, B, offsetVec, offsetDist):
    
    fScaled = (f - f[-1])

    x = offsetVec[:,0]
    y = offsetVec[:,1]
    
    # compute least squares coefficients for quadratic cost surface
    K = np.matmul(B, fScaled[0:-1])
    
    # compute eigen values of the surface
    a = K[3]
    c = K[2]/2
    b = K[4]
    
    t1 = 0.5 * (a + b)
    t2 = 0.5 * np.sqrt((a-b)**2 + 4*c*c)
    eig1 = t1 + t2
    eig2 = t1 - t2
    
    # if the quadratic is positive definite, and the center point is the minimum,
    # sovlve for the minimum of the quadratic
    if (eig1 > 0 and eig2 > 0) and (fScaled.min() > 0):
        
        Btmp = np.array([-K[0], -K[1]])
        Atmp = np.array([[2*K[3], K[2]], [K[2], 2*K[4]]])
        
        minLoc = np.linalg.solve(Atmp, Btmp)
 
    # otherwise, chose the minumum sample as the next step       
    else:
        #gradDecent = -K[[0,1]]
        indexMin = np.argmin(fScaled)
        minLoc = offsetVec[indexMin, :]

    # limit the magnitude of travel from the current location
    minLocNorm = np.sqrt(minLoc[0]*minLoc[0] + minLoc[1]*minLoc[1])
    if minLocNorm > offsetDist:
        minLoc = offsetDist * minLoc / minLocNorm
        
    # round to an integer number of pixels
    deltaPixels = np.floor(minLoc).astype(int)
        
    return deltaPixels

#--------------------------------------------------------------------------
# l - length of window
def computeRgbMatShiftError(rgbMat1, rgbMat2, zMat1, zMat2, mask1, mask2, l, index1, index2, offset):
    
    # the first image
    ir1 = index1[0]
    ic1 = index1[1]
    
    # the image we are trying to find a match to
    ir2 = index2[0] + offset[0]
    ic2 = index2[1] + offset[1]
    
    mask = np.logical_and(mask1[ir1:ir1+l, ic1:ic1+l], mask2[ir2:ir2+l, ic2:ic2+l])
    
    fr = computeMatError(rgbMat1[ir1:ir1+l, ic1:ic1+l, 0], rgbMat2[ir2:ir2+l, ic2:ic2+l, 0], mask)
    fg = computeMatError(rgbMat1[ir1:ir1+l, ic1:ic1+l, 1], rgbMat2[ir2:ir2+l, ic2:ic2+l, 1], mask)
    fb = computeMatError(rgbMat1[ir1:ir1+l, ic1:ic1+l, 2], rgbMat2[ir2:ir2+l, ic2:ic2+l, 2], mask)
    fz = computeMatError(zMat1[ir1:ir1+l, ic1:ic1+l], zMat2[ir2:ir2+l, ic2:ic2+l], mask)
    
    zScale = 0
    
    return fr + fg + fb + zScale * fz


#------------------------------------------------------------------------------
def normalizeEntropymat(Hmat, Hmask, lc, uc):
    
    Hflat = Hmat.flatten()
    HmaskFlat = Hmask.flatten()

    H2 = Hflat[HmaskFlat]
    H2var = np.sqrt(np.var(H2))
    H2mean = np.mean(H2)
    H = (Hmat - H2mean) / H2var
    
    H[H < lc] = lc
    H[H > uc] = uc

    H = (H - lc) / (uc - lc)
    
    H[np.invert(Hmask)] = 0
    
    return H
    
#------------------------------------------------------------------------------
def rgbToGreyMat(rgbMat):
    return 0.299*rgbMat[:,:,0] + 0.587*rgbMat[:,:,1] + 0.114*rgbMat[:,:,2]


# ~~ myCv class ~~
class myCv:
    #--------------------------------------------------------------------------
    # construction
    def __init__(self, height, width):
  
        self.width = width
        self.height = height
        
        # the gaussian kernel for image blurring (I know this is a seperable filter, but that's more code to write)
        self.gK = np.array([[1, 4,  7,  4,  1],
                            [4, 16, 26, 16, 4],
                            [7, 26, 41, 26, 7],
                            [4, 16, 26, 16, 4],
                            [1, 4,  7,  4,  1]]); 
        
        self.uK = np.ones((5,5))
        
        self.LoG = np.array([[0,  1,  1,  2,  2,    2,   1,  1,  0],
                             [1,  2,  4,  5,  5,    5,   4,  2,  1],
                             [1,  4,  5,  3,  9,    3,   5,  4,  1],
                             [2,  5,  3, -12, -24, -12,  3,  5,  2],
                             [2,  5,  0, -24, -40, -24,  0,  5,  2],
                             [2,  5,  3, -12, -24, -12,  3,  5,  2],
                             [1,  4,  5,  3,  9,    3,   5,  4,  1],
                             [1,  2,  4,  5,  5,    5,   4,  2,  1],
                             [0,  1,  1,  2,  2,    2,   1,  1,  0]])
        
        # Sobel operator kernel
        self.gx = np.array([[1,  0, -1],
                            [2,  0, -2],
                            [1,  0, -1]])
        
        self.gy = np.array([[1,   2,  1],
                            [0,   0,  0],
                            [-1, -2, -1]])
        
        self.nBins = 128

    #--------------------------------------------------------------------------
    # methods
    def computeMaskMat(self, mat):
        
        maskMat = np.ones((self.height, self.width))
        maskMat[mat == 0] = 0
                       
        return maskMat
    
    
    #--------------------------------------------------------------------------
    def kernelConv(self, mat, maskMat, K):
        
        height, width = mat.shape
        h, w = K.shape

        dim = np.floor(h/2).astype(np.int)   
        blurMat = np.zeros((height, width))
        
        for ii in range(dim, height-dim):
            for jj in range(dim, width-dim):
                
                if (maskMat[ii,jj] == 1):
                    
                    gKm, sumWeight = conv(K, maskMat, ii, jj, dim)
                    gKm, sumKernel = conv(gKm, mat, ii, jj, dim)
                    
                    if sumWeight > 0:
                        blurMat[ii,jj] = sumKernel / sumWeight
                        
        return blurMat  


    #--------------------------------------------------------------------------
    def blurMat(self, mat, maskMat):  
        return self.kernelConv(mat, maskMat, self.gK)
    
    #--------------------------------------------------------------------------
    def LoGMat(self, mat, maskMat):  
        return self.kernelConv(mat, maskMat, self.LoG)    
    

    #--------------------------------------------------------------------------
    def estimateRgbPdf(self, redMat, greenMat, blueMat, maskMat):   
        
        pdf = np.zeros((self.nBins, self.nBins, self.nBins))
        sm = 0
        
        for ii in range(self.height):
            for jj in range(self.width):
                
                if maskMat[ii,jj]:
                    
                    idr = np.floor(redMat[ii,jj] / 2).astype(int)
                    idg = np.floor(greenMat[ii,jj] / 2).astype(int)
                    idb = np.floor(blueMat[ii,jj] / 2).astype(int)
                    
                    pdf[idr,idg,idb] += 1
                    sm += 1
                    
        pdfNormed = pdf / sm
    
        return pdfNormed
    
    #--------------------------------------------------------------------------
    def computeImageEntropy(self, redMat, greenMat, blueMat, maskMat):  
        
        (h, w) = redMat.shape
        
        X = np.zeros((3, h*w))
        X[0,:] = redMat.flatten()
        X[1,:] = greenMat.flatten()
        X[2,:] = blueMat.flatten()
        mask   = maskMat.flatten()
        
        #mask = np.abs(np.sum(X, axis = 0)) > 0
        N = mask.sum()    
        H = 0
        Hvalid = False
        
        # need atleast a few points to compute mean
        if N > 3:
            X = X[:, mask]
            X = X - X.mean(axis=1, keepdims=True)
            
            A = np.matmul(X, X.T) / (N-1)
            #expH = np.linalg.det(A)
            expH = det3x3Symetric(A)
            
            if expH > 0:
                H = np.log(expH) 
                Hvalid = True
        
        return H, Hvalid
    
    #--------------------------------------------------------------------------
    # coarse grid the image, using a threshold on the number of pixels that are
    # valid within the sub-block area
    def coarseGridEntropy(self, redMat, greenMat, blueMat, maskMat, l):
    
        w = np.floor(self.width/l).astype(np.int)
        h = np.floor(self.height/l).astype(np.int)
        
        gridHMat = np.zeros((h, w))
        gridHMaskMat = np.full((h, w), False, dtype = bool)
        
        # use a buffer of 1 grid block length around edge of image
        for rowId, ii in enumerate(range(l, self.height-l, l)):
            for colId, jj in enumerate(range(l, self.width-l, l)):
            
                rM =   redMat[ii:ii+l, jj:jj+l]
                gM = greenMat[ii:ii+l, jj:jj+l]
                bM =  blueMat[ii:ii+l, jj:jj+l]
                mM =  maskMat[ii:ii+l, jj:jj+l]
                
                gridHMat[rowId+1,colId+1], gridHMaskMat[rowId+1,colId+1] = self.computeImageEntropy(rM, gM, bM, mM)     
                        
        return gridHMat, gridHMaskMat    
            
    
    #--------------------------------------------------------------------------
    def computeWindowedImageEntropy(self,  rgbMat, maskMat): 
        
        Hmat = np.zeros((self.height, self.width))
        Hmask = np.full((self.height, self.width), False, dtype = bool)
        win = 4
        for ii in range(self.height):
            for jj in range(self.width):
                
                if (maskMat[ii,jj] == 1):
                    
                    ac = np.max((0, ii - win))
                    bc = np.min((self.height - 1, ii + win + 1)) 
                    
                    ar = np.max((0, jj - win))
                    br = np.min((self.width - 1, jj + win + 1)) 
                    
                    rM = rgbMat[ac:bc, ar:br, 0]
                    gM = rgbMat[ac:bc, ar:br, 1]
                    bM = rgbMat[ac:bc, ar:br, 2]
                    mM = maskMat[ac:bc, ar:br]
                    
                    Hmat[ii,jj], Hmask[ii,jj] = self.computeImageEntropy(rM, gM, bM, mM)
                        
        return Hmat, Hmask
    
    

        
        
        
        
        