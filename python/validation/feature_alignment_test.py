import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import sys
import open3d as o3d

 # custom functions
sys.path.insert(1, '../functions')
import video_functions as vid
import feature_detection_functions as fd
import feature_detection_functions_gpu as fdgpu
import feature_matching_functions_gpu as fmgpu
import cv_functions as cvFun
import harris_detector_functions_gpu as hd
import cv_functions_gpu as cvGpu

loadData = True
plt.close('all')

#------------------------------------------------------------------------------
# helper functions
#------------------------------------------------------------------------------
def plot_feature(ax, featMats, idx, scale):

    mat = np.stack((np.reshape(featMats['feat_rMat'][idx,:], (scale, scale)),
                    np.reshape(featMats['feat_gMat'][idx,:], (scale, scale)),
                    np.reshape(featMats['feat_bMat'][idx,:], (scale, scale))), axis = 2)
    ax.imshow(mat.get()/255)
    
def get_frame_data(frame_id, redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens):
    
    rgbMat, xyzMat, maskMat = vid.getFrameMats(redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens, frame_id)   
    
    gpuDict = {
    'rMat':    cp.array(rgbMat[:,:,0], dtype = cp.float32),
    'gMat':    cp.array(rgbMat[:,:,1], dtype = cp.float32),
    'bMat':    cp.array(rgbMat[:,:,2], dtype = cp.float32),
    'greyMat': cp.array(cvFun.rgbToGreyMat(rgbMat.astype(float)), dtype = cp.float32),
    'maskMat': cp.array(maskMat, dtype = bool)  }
    
    return gpuDict, rgbMat

def blur_frame_data(gpuDict):
    
    blurObj = cvGpu.gaussianBlur()
    
    gpuDict_blurred = {
    'rMat':    blurObj.return_blurImg(gpuDict['rMat'],    gpuDict['maskMat']),
    'gMat':    blurObj.return_blurImg(gpuDict['gMat'],    gpuDict['maskMat']),
    'bMat':    blurObj.return_blurImg(gpuDict['bMat'],    gpuDict['maskMat']),
    'greyMat': gpuDict['greyMat'],
    'maskMat': gpuDict['maskMat'] }
    
    return gpuDict_blurred
 
class localMinimizer():
    def __init__(self):
        self.offsetMat = np.array([[-1, 0, 1,-1, 0, 1,-1, 0, 1],
                                   [-1,-1,-1, 0, 0, 0, 1, 1, 1]])
        nP = self.offsetMat.shape[1]
        A = np.zeros((nP, 5))
        for offset in range(nP):
            x = self.offsetMat[0, offset]
            y = self.offsetMat[1, offset]
            A[offset, :] = np.array([x, y, x*y, x*x, y*y])
    
        self.B = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        
    def find_min(self, C):
        
        #f = C[self.offsetMat[1,:], self.offsetMat[0,:]]
        f = C.flatten()
        f = f - f.min()
        
        # compute least squares coefficients for quadratic cost surface
        K = np.matmul(self.B, f)
        
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
        print('eigen value ratio')
        print(eig1 / eig2)
        if (eig1 > 0 and eig2 > 0):
            
            Btmp = np.array([-K[0], -K[1]])
            Atmp = np.array([[2*K[3], K[2]], [K[2], 2*K[4]]])
            
            minLoc = np.linalg.solve(Atmp, Btmp)
     
        # otherwise, chose the minumum sample as the next step       
        else:
            print('warning not positive definite')
            minLoc = np.array([0, 0])
        return  minLoc
            
        
def computeMinCostLocation(C):
    
    h, w = C.shape
    x_mid = np.int32((w-1)/2)
    y_mid = np.int32((h-1)/2)
    
    minId = np.unravel_index(C.argmin(), C.shape)
    y_min = minId[0]
    x_min = minId[1]
    delta_y = y_min - y_mid
    delta_x = x_min - x_mid
    
    if (x_min-1 > 0) and (x_min+1 < w) and (y_min-1 > 0) and (y_min+1 < h):
    
        C_min = C[y_min-1:y_min+2, x_min-1:x_min+2]
        print(C_min)
        localMinObj = localMinimizer()
        dx, dy = localMinObj.find_min(C_min)
        
        valid = True
        
    else:
        y_min = y_mid
        x_min = x_mid
        dx, dy = 0, 0
        valid = False
        
    x_min_loc = x_min + dx
    y_min_loc = y_min + dy
    
    dx_min = delta_x + dx
    dy_min = delta_y + dy
   
    return x_min_loc, y_min_loc, dx_min, dy_min, valid
    
    
#------------------------------------------------------------------------------
# data configuration and loading
#------------------------------------------------------------------------------
if loadData or 'redTens' not in locals():
     # calibration data
    folder = '../../data/'
    calName = folder + 'calibration.h5'
    numpyName = folder + 'rawData.npz'
         
    # video data
    start = time.time()
    redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens = vid.loadDataSet(calName, numpyName, folder)
    print('timer:', time.time() - start)
    
# load two different frames
(height, width, nFrms) = redTens.shape
print('number of frames: ', nFrms)

# create object for handling CV functions
myCv = cvFun.myCv(height, width) 

# get frame 1 mats
frame1 = 30
frame2 = frame1 + 8

gpuDat1, rgbMat1 = get_frame_data(frame1, redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens)
gpuDat2, rgbMat2 = get_frame_data(frame2, redTens, greenTens, blueTens, xTens, yTens, zTens, maskTens)

height, width = gpuDat1['maskMat'].shape
    
#------------------------------------------------------------------------------
# course feature matching
#------------------------------------------------------------------------------

cornerScale = 4
matchScale = 15
nFeatures = 128

# corner detector object 
#cornerObjGpu = fdgpu.corner_detector_class(height, width, cornerScale, nFeatures)
cornerObjGpu = hd.HarrisDetectorGpu(height, width, cornerScale, nFeatures)

# matching object
matchObjGpu = fmgpu.feature_matching_class(height, width, nFeatures, nFeatures, matchScale)

# corner points
cornerPointIdx1_gpu = cp.zeros((2,nFeatures), dtype = cp.int32)
cornerPointIdx2_gpu = cp.zeros((2,nFeatures), dtype = cp.int32)
cornerObjGpu.findCornerPoints(cornerPointIdx1_gpu, gpuDat1['greyMat'], gpuDat1['maskMat'].astype(cp.float32))
cornerObjGpu.findCornerPoints(cornerPointIdx2_gpu, gpuDat2['greyMat'], gpuDat2['maskMat'].astype(cp.float32))

# find matches
matchObjGpu.set_img1_features(gpuDat1['rMat'], gpuDat1['gMat'], gpuDat1['bMat'], gpuDat1['maskMat'], cornerPointIdx1_gpu)
matchObjGpu.set_img2_features(gpuDat2['rMat'], gpuDat2['gMat'], gpuDat2['bMat'], gpuDat2['maskMat'], cornerPointIdx2_gpu)
cornerMatchedIdx1, cornerMatchedIdx2 = matchObjGpu.computeFeatureMatches()

# re-set the features to be matched pairs
matchObjGpu.set_img1_features(gpuDat1['rMat'], gpuDat1['gMat'], gpuDat1['bMat'], gpuDat1['maskMat'], cornerMatchedIdx1)
matchObjGpu.set_img2_features(gpuDat2['rMat'], gpuDat2['gMat'], gpuDat2['bMat'], gpuDat2['maskMat'], cornerMatchedIdx2)

cornerMatchIdx1 = cornerMatchedIdx1.get()
cornerMatchIdx2 = cornerMatchedIdx2.get()
#cornerMatchIdx1 = cornerPointIdx1_gpu.get()
#cornerMatchIdx2 = cornerPointIdx2_gpu.get()

n_matches = cornerMatchIdx1.shape[1]
print('number of frames matched:')
print(n_matches)

# plotting 
rgb1Match = np.copy(rgbMat1)
rgb2Match = np.copy(rgbMat2)

for ii in range(cornerMatchIdx1.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchIdx1[0, ii], cornerMatchIdx1[1, ii], 12, color.astype(np.ubyte))
    fd.drawBox(rgb2Match, cornerMatchIdx2[0, ii], cornerMatchIdx2[1, ii], 12, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points')
plt.title('rgb  frame 1 interest points')
plt.imshow(rgb1Match)

plt.figure('rgb  frame 2 interest points')
plt.title('rgb  frame 2 interest points')
plt.imshow(rgb2Match)


#------------------------------------------------------------------------------
# fine feature alignment
#------------------------------------------------------------------------------

# dimension of box the check for best match in
d_box = matchScale
nFeatures1 = 1
nFeatures2 = d_box * d_box
feat_id = 15

alignObjGpu = fmgpu.feature_matching_class(height, width, nFeatures1, nFeatures2, matchScale)

for feat_id in range(n_matches):
    corner1_id = cornerMatchIdx1[:,feat_id][:,None]
    corner2_id = cornerMatchIdx2[:,feat_id][:,None]
    
    offset = np.int32((d_box-1)/2)
    offset_x, offset_y = np.meshgrid(np.arange(0, d_box), np.arange(0, d_box))
    offset_x -= offset
    offset_y -= offset
    
    offset_vec = np.stack((offset_x.flatten(), offset_y.flatten()))
    offset_id = corner2_id + offset_vec 
    
    corner1_id = cp.array(corner1_id, dtype = cp.int32)
    offset_id = cp.array(offset_id, dtype = cp.int32)
    
    alignObjGpu.set_img1_features(gpuDat1['rMat'], \
                                  gpuDat1['gMat'], \
                                  gpuDat1['bMat'], \
                                  gpuDat1['maskMat'], \
                                  corner1_id)
        
    alignObjGpu.set_img2_features(  gpuDat2['rMat'], \
                                    gpuDat2['gMat'], \
                                    gpuDat2['bMat'], \
                                    gpuDat2['maskMat'], \
                                    offset_id)
          
    _, _ = alignObjGpu.computeFeatureMatches()
    c2 = alignObjGpu.costMat.get()
    C_align = np.reshape(c2, (d_box, d_box), order='F')
    
    x_min_loc, y_min_loc, dx_min, dy_min, valid  = computeMinCostLocation(C_align)
    print(f'delta x pixels: {dx_min}')
    print(f'delta y pixels: {dy_min}')
    print(f'valid sample: {valid}')
    
    cornerMatchIdx1[0,feat_id] = cornerMatchIdx1[0,feat_id] - np.round(dy_min).astype(int)
    cornerMatchIdx1[1,feat_id] = cornerMatchIdx1[1,feat_id] - np.round(dx_min).astype(int)
    
    if False:
        fig, ax = plt.subplots(1,3)
        plot_feature(ax[0], matchObjGpu.im1_featMats, feat_id, matchScale)
        plot_feature(ax[1], matchObjGpu.im2_featMats, feat_id, matchScale)
        ax[2].imshow(C_align)
        ax[2].plot(x_min_loc, y_min_loc, 'ro')
        

# plotting 
rgb1Match = np.copy(rgbMat1)
for ii in range(cornerMatchIdx1.shape[1]):
    
    color = 255 * np.random.rand(3)
    fd.drawBox(rgb1Match, cornerMatchIdx1[0, ii], cornerMatchIdx1[1, ii], 12, color.astype(np.ubyte))
    
plt.figure('rgb  frame 1 interest points aligned')
plt.title('rgb  frame 1 interest points aligned')
plt.imshow(rgb1Match)

