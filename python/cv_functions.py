import numpy as np

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
def computeMatShiftError(mat1, mat2, l, index, offset):
    
    ir1 = index[0]
    ic1 = index[1]
    ir2 = index[0] + offset[0]
    ic2 = index[1] + offset[1]
    
    matErr = mat1[ir1:ir1+l, ic1:ic1+l] - mat2[ir2:ir2+l, ic2:ic2+l]
    f = np.sum(matErr * matErr)
    
    return f

#--------------------------------------------------------------------------
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
        
        h, w = K.shape
        dim = np.floor(h/2).astype(np.int)   
        blurMat = np.zeros((self.height, self.width))
        
        for ii in range(dim, self.height-dim):
            for jj in range(dim, self.width-dim):
                
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
    # coarse grid the image, using a threshold on the number of pixels that are
    # valid within the sub-block area
    def gridSpace(self, maskMat, l, thresh):
    
        w = np.floor(self.width/l).astype(np.int)
        h = np.floor(self.height/l).astype(np.int)
        gridMat = np.full((h, w), False, dtype = bool)
        gridFullMat = np.full((self.height, self.width), False, dtype = bool)
        
        for rowId, ii in enumerate(range(0, self.height, l)):
            for colId, jj in enumerate(range(0, self.width, l)):
                
                sumMask = np.sum(maskMat[ii:ii+l, jj:jj+l]).astype(int)
                if (sumMask >= thresh):
                    gridMat[rowId, colId] = True
                    gridFullMat[ii:ii+l, jj:jj+l] = True
                    
                        
        return gridMat, gridFullMat
    

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
    def computeImageEntropy(self,  redMat, greenMat, blueMat, maskMat): 
        
        Hmat = np.zeros((self.height, self.width))
        Hmask = np.full((self.height, self.width), False, dtype = bool)
        win = 3
        Nlim = 40
        for ii in range(self.height):
            for jj in range(self.width):
                
                if (maskMat[ii,jj] == 1):
                    
                    ac = np.max((0, ii - win))
                    bc = np.min((self.height - 1, ii + win + 1)) 
                    
                    ar = np.max((0, jj - win))
                    br = np.min((self.width - 1, jj + win + 1)) 
                    
                    rM =   redMat[ac:bc, ar:br]
                    gM = greenMat[ac:bc, ar:br]
                    bM =  blueMat[ac:bc, ar:br]
                    mM =  maskMat[ac:bc, ar:br]
                    
                    (h, w) = rM.shape
                    
                    X = np.zeros((3, h*w))
                    X[0,:] = rM.flatten()
                    X[1,:] = gM.flatten()
                    X[2,:] = bM.flatten()
                    
                    mask = np.abs(np.sum(X, axis = 0)) > 0
                    N = mask.sum()
                    
                    if N > Nlim:
                        
                        X = X[:, mask]
                        X = X - X.mean(axis=1, keepdims=True)
                        
                        A = np.matmul(X, X.T) / (N-1)
                        #h = np.linalg.det(A)
                        h = det3x3Symetric(A)
                        
                        if h > 0:
                                
                            Hmat[ii,jj] = np.log(h) 
                            Hmask[ii,jj] = True
                        
        return Hmat, Hmask
    
    

        
        
        
        
        