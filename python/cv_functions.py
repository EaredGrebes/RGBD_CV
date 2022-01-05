import numpy as np

# ~~ helper functions ~~    
#------------------------------------------------------------------------------
def conv5x5(K, Mat, iRow, iCol):
    
    gKm = K * Mat[iRow-2:iRow+3, iCol-2:iCol+3]
    sumK = np.sum(gKm)
    
    return gKm, sumK


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
    def blurMat(self, mat, maskMat):
        
        blurMat = np.zeros((self.height, self.width))

        for ii in range(2, self.height-2):
            for jj in range(2, self.width-2):
                
                if (maskMat[ii,jj] == 1):
                
                    gKm, sumWeight = conv5x5(self.gK, maskMat, ii, jj)
                    gKm, sumKernel = conv5x5(gKm, mat, ii, jj)
                    
                    if sumWeight > 0:
                        blurMat[ii,jj] = sumKernel / sumWeight
                        
        return blurMat  
    
    
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
    def computeImageEntropy(self, rgbPdf, redMat, greenMat, blueMat, maskMat): 
        
        H = np.zeros((self.height, self.width))
        
        for ii in range(2, self.height-2):
            for jj in range(2, self.width-2):
                
                if (maskMat[ii,jj] == 1):
                    
                    rSub = redMat[iRow-2:iRow+3, iCol-2:iCol+3]
                    gSub = greenMat[iRow-2:iRow+3, iCol-2:iCol+3]
                    bSub = blueMat[iRow-2:iRow+3, iCol-2:iCol+3]
                    maskSub = maskMat[iRow-2:iRow+3, iCol-2:iCol+3]
                    
                    rgbPdf = self.estimateRgbPdf(rSub, gSub, bSub, maskSub)
                    
                    idr = np.floor(redMat[ii,jj] / 2).astype(int)
                    idg = np.floor(greenMat[ii,jj] / 2).astype(int)
                    idb = np.floor(blueMat[ii,jj] / 2).astype(int)
                    
                    p = rgbPdf[idr, idg, idb]
                    
                    H[ii,jj] = p * np.log2(1/p)
    
        return H    