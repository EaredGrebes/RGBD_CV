import cv_functions as cvFun
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------

def quadraticSurface(f, B, offsetVec):
    
    # last point is the center point
    fScaled = (f - f[-1])

    x = offsetVec[:,0]
    y = offsetVec[:,1]
    
    # compute least squares coefficients for quadratic cost surface
    K = np.matmul(B, fScaled[0:-1])
    
    gradient = K[[0,1]]
    
    # compute eigen values of the quadratic surface
    a = K[3]
    c = K[2]/2
    b = K[4]
    D = np.sqrt((a-b)**2 + 4*c*c)
    
    t1 = 0.5 * (a + b)
    t2 = 0.5 * D
    eigVal1 = t1 + t2
    eigVal2 = t1 - t2
    
    if eigVal1 > eigVal2:
        eigVec = np.array([1,  b - a  + D])
    else:
        eigVec = np.array([1, b - a - D])
    
    eigVec = eigVec / np.sqrt(eigVec[0]**2 + eigVec[1]**2)
    eigVec = np.abs(eigVec)
    
    if eigVec[0] > 1e-3:
        angle = np.atan(eigVec[1] / eigVec[0])
    else:
        angle = np.pi/2
        
    return angle, eigVal1, eigVal2, gradient


#------------------------------------------------------------------------------

offsetMat0 = np.array([])

nP, _ = offsetMat0.shape  # number of finite difference points  

# create least squares matrix for computing quadratic model of cost function from offset error points  
A = np.zeros((nP-1, 5))
for offset in range(nP-1):
    x = offsetMat[offset, 0]
    y = offsetMat[offset, 1]
    
    A[offset, :] = np.array([x, y, x*y, x*x, y*y])

B = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        