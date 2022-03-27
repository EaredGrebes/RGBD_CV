#include <math_constants.h>
#include <opencv2/cudaarithm.hpp>
#include "cuda_math_helpers.h"
#include "svd3_cuda.h"


///////////////////////////////////////////////////////////////////////////////
// constants (for development, maybe move somewhere else in the future)
///////////////////////////////////////////////////////////////////////////////

// for the color projections (848 x 480)
//                                      ppx      ppx      fx       fy
__constant__ float rgbPixelCoeffs[4] = {420.779, 244.096, 421.258, 420.961};

// these are for correcting the rgb lens distortion, equation and coefficients taken from realsense api
__constant__ float rgbLensCoeffs[5] = {-0.0554403, 0.0637236, -0.000739898, 0.000511257, -0.0202138};

//                                        ppx      ppy      fx       fy
__constant__ float depthPixelCoeffs[4] = {423.337, 238.688, 421.225, 421.225};

//  should be zero, see this:
// https://github.com/IntelRealSense/librealsense/issues/1430
__constant__ float depthLensCoeffs[5] = {0, 0, 0, 0, 0};

// kalman filter parameters 
__constant__ float processNoiseVar_mm2 = 200.0f;
__constant__ float measNoiseVar_mm2 = 100.0f;

__constant__ float depthVarMax_mm2 = 5000 * 5000;
__constant__ float depthVarMin_mm2 = 4;

__constant__ float depthConvergedThresh_mm2 = 400.0f;

// this coresponds to throwing out measurements where the camera line of sight 
// for a given pixel is 80 deg off from the estimated normal of that pixel
// cos(80 deg) = 0.17
__constant__ float angleThresh = 0.17f;

__constant__ float k_ave = 0.3f;   

// the gaussian kernel for image blurring (I know this is a seperable filter, but that's more code to write)
__constant__ int gK[25] = {1, 4,  7,  4,  1,
                           4, 16, 26, 16, 4,
                           7, 26, 41, 26, 7,
                           4, 16, 26, 16, 4,
                           1, 4,  7,  4,  1}; 

__constant__ int uK[25] = {1, 1,  1,  1,  1,
                           1, 1,  1,  1,  1,
                           1, 1,  1,  1,  1,
                           1, 1,  1,  1,  1,
                           1, 1,  1,  1,  1}; 

// Sobel operator kernel
__constant__ int gx[9] = {1, 0, -1,
                          2, 0, -2,
                          1, 0, -1};

__constant__ int gy[9] = {1,  2,  1,
                          0,  0,  0,
                         -1, -2, -1};

 __constant__ float gradScale = 4.0e-3;                      


///////////////////////////////////////////////////////////////////////////////
// cuda device functions.  only called from kernel (__global__) functions
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// this function takes a 3d x,y,z point and computes it's pixel x,y integer index
__device__ __forceinline__ bool projectPointToPixel(uint2 &pixelXY,         // outputs
                                                    float3 &position_m,     // inputs
                                                    float *pixelCoeffs,
                                                    float *lensCoeffs,
                                                    unsigned int width,
                                                    unsigned int height)
{

    bool pixelInRange = true;
    float xPixel = position_m.x / position_m.z;
    float yPixel = position_m.y / position_m.z;

    float r2  = xPixel*xPixel + yPixel*yPixel;
    float f = 1 + lensCoeffs[0]*r2 + lensCoeffs[1]*r2*r2 + lensCoeffs[4]*r2*r2*r2;
    xPixel *= f;
    yPixel *= f;
    float dx = xPixel + 2*lensCoeffs[2]*xPixel*yPixel + lensCoeffs[3]*(r2 + 2*xPixel*xPixel);
    float dy = yPixel + 2*lensCoeffs[3]*xPixel*yPixel + lensCoeffs[2]*(r2 + 2*yPixel*yPixel);
    xPixel = dx;
    yPixel = dy;

    //int pixelColId = __float2int_rn(xPixel * fx + ppx);
    //int pixelRowId = __float2int_rn(yPixel * fy + ppy);
    int pixelColId = __float2int_rn(xPixel * pixelCoeffs[2] + pixelCoeffs[0]);
    int pixelRowId = __float2int_rn(yPixel * pixelCoeffs[3] + pixelCoeffs[1]);

    if (pixelRowId < 0){
        pixelRowId = 0;
        pixelInRange = false;
    }
    else if (pixelRowId > (height - 1)){
        pixelRowId = (height - 1);
        pixelInRange = false;
    }
    if (pixelColId < 0){
        pixelColId = 0;
        pixelInRange = false;
    }
    else if (pixelColId > (width - 1)){
        pixelColId = (width - 1);
        pixelInRange = false;
    }

    pixelXY = make_uint2(pixelColId, pixelRowId);

    return pixelInRange;
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ int Conv5x5(int *gKm,                         // outputs
                                       cv::cuda::PtrStepSz<int> matPtr,  // inputs
                                       int *K,
                                       int iRow,
                                       int iCol)
{
    // loops unrolled, because I read multiple places this is faster for Cuda, as the nvcc compiler does not unroll them
    gKm[0]  = K[0]*matPtr(iRow-2, iCol-2);
    gKm[1]  = K[1]*matPtr(iRow-2, iCol-1);
    gKm[2]  = K[2]*matPtr(iRow-2, iCol);
    gKm[3]  = K[3]*matPtr(iRow-2, iCol+1);
    gKm[4]  = K[4]*matPtr(iRow-2, iCol+2);
    gKm[5]  = K[5]*matPtr(iRow-1, iCol-2);
    gKm[6]  = K[6]*matPtr(iRow-1, iCol-1);
    gKm[7]  = K[7]*matPtr(iRow-1, iCol);
    gKm[8]  = K[8]*matPtr(iRow-1, iCol+1);
    gKm[9]  = K[9]*matPtr(iRow-1, iCol+2);
    gKm[10] = K[10]*matPtr(iRow,  iCol-2);
    gKm[11] = K[11]*matPtr(iRow,  iCol-1);
    gKm[12] = K[12]*matPtr(iRow,  iCol);
    gKm[13] = K[13]*matPtr(iRow,  iCol+1);
    gKm[14] = K[14]*matPtr(iRow,  iCol+2), 
    gKm[15] = K[15]*matPtr(iRow+1, iCol-2);
    gKm[16] = K[16]*matPtr(iRow+1, iCol-1);
    gKm[17] = K[17]*matPtr(iRow+1, iCol);
    gKm[18] = K[18]*matPtr(iRow+1, iCol+1);
    gKm[19] = K[19]*matPtr(iRow+1, iCol+2);
    gKm[20] = K[20]*matPtr(iRow+2, iCol-2);
    gKm[21] = K[21]*matPtr(iRow+2, iCol-1);
    gKm[22] = K[22]*matPtr(iRow+2, iCol);
    gKm[23] = K[23]*matPtr(iRow+2, iCol+1);
    gKm[24] = K[24]*matPtr(iRow+2, iCol+2);

    int sumK = gKm[0]  + gKm[1]  + gKm[2]  + gKm[3] + gKm[4] + gKm[5] + gKm[6] + gKm[7] + gKm[8] + gKm[9] + gKm[10] + gKm[11] + gKm[12] + gKm[13] + gKm[14] + gKm[15] 
             + gKm[16] + gKm[17] + gKm[18] + gKm[19]+ gKm[20]+ gKm[21]+ gKm[22]+ gKm[23]+ gKm[24];

    return sumK;                 
}


///////////////////////////////////////////////////////////////////////////////
// Kernel functions
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// this implements a guassian blur for integer data type images
// why int? because cuda atomicMin cannot use floating point values,
// and the depth image needs atomicMin for shading
__global__ void GaussianBlur5x5Kernel(cv::cuda::PtrStepSz<int> matBlurredPtr, // outputs
                                      cv::cuda::PtrStepSz<int> matPtr,        // inputs
                                      cv::cuda::PtrStepSz<int> cMatPtr,
                                      unsigned int width,
                                      unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    if (cMatPtr(iRow, iCol) == 1) {

        if ( (iCol >= 2)           &&
             (iCol < (width - 2))  &&
             (iRow >= 2)           &&
             (iRow < (height - 2))) {

            int gKm[25];
            int sumWeight = Conv5x5(gKm, cMatPtr, gK, iRow, iCol);
            int tmp[25];
            int sumKernel = Conv5x5(tmp, matPtr, gKm, iRow, iCol);
        
            if (sumWeight > 0)
            {
                matBlurredPtr(iRow, iCol) = __float2int_rn(__int2float_rn(sumKernel) / __int2float_rn(sumWeight));
            } else {
                matBlurredPtr(iRow, iCol) = 0;
            }
        }

    } else {
        matBlurredPtr(iRow, iCol) = matPtr(iRow, iCol);
    }
}


//-----------------------------------------------------------------------------
// fills in missing color data in when there is enough adjacent valid color samples
// some color pixels are not available because of the shading effects that occur when translating
// the color camera pixels into the depth camera reference frame
__global__ void fillInColorKernel(  cv::cuda::PtrStepSz<int> clrBlurMatPtr_r, // outputs
                                    cv::cuda::PtrStepSz<int> clrBlurMatPtr_g,
                                    cv::cuda::PtrStepSz<int> clrBlurMatPtr_b,
                                    cv::cuda::PtrStepSz<int> maskMatBlurPtr,
                                    cv::cuda::PtrStepSz<int> maskMatPtr,      // inputs
                                    cv::cuda::PtrStepSz<int> clrMatPtr_r,   
                                    cv::cuda::PtrStepSz<int> clrMatPtr_g,
                                    cv::cuda::PtrStepSz<int> clrMatPtr_b,
                                    int thresh,
                                    unsigned int width,
                                    unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    clrBlurMatPtr_r(iRow, iCol) = clrMatPtr_r(iRow, iCol);
    clrBlurMatPtr_g(iRow, iCol) = clrMatPtr_g(iRow, iCol);
    clrBlurMatPtr_b(iRow, iCol) = clrMatPtr_b(iRow, iCol);

    if (maskMatPtr(iRow, iCol) == 0) {

        if ( (iCol >= 2)           &&
             (iCol < (width - 2))  &&
             (iRow >= 2)           &&
             (iRow < (height - 2))) {

            int gKm[25];
            int tmp[25];
            int uKm[25];
            int sumMask   = Conv5x5(uKm, maskMatPtr, uK, iRow, iCol);
            int sumWeight = Conv5x5(gKm, maskMatPtr, gK, iRow, iCol);

            int sumKernel_r = Conv5x5(tmp, clrMatPtr_r, gKm, iRow, iCol);
            int sumKernel_g = Conv5x5(tmp, clrMatPtr_g, gKm, iRow, iCol);
            int sumKernel_b = Conv5x5(tmp, clrMatPtr_b, gKm, iRow, iCol);
        
            // if 
            if (sumMask > thresh) {

                clrBlurMatPtr_r(iRow, iCol) = __float2int_rn(__int2float_rn(sumKernel_r) / __int2float_rn(sumWeight));
                clrBlurMatPtr_g(iRow, iCol) = __float2int_rn(__int2float_rn(sumKernel_g) / __int2float_rn(sumWeight));
                clrBlurMatPtr_b(iRow, iCol) = __float2int_rn(__int2float_rn(sumKernel_b) / __int2float_rn(sumWeight));
            } else {
                // mark the pixel index where there wasn't enough data to fill in the color
                maskMatBlurPtr(iRow, iCol) = 0;

                clrBlurMatPtr_r(iRow, iCol) = 0;
                clrBlurMatPtr_g(iRow, iCol) = 0;
                clrBlurMatPtr_b(iRow, iCol) = 0;
            }
        }
    } 
}


//-----------------------------------------------------------------------------
__global__ void edgeDetectorKernel(cv::cuda::PtrStepSz<int> edgeMaskMat,   // outputs
                                   cv::cuda::PtrStepSz<int> maskMatPtr,    // inputs
                                   cv::cuda::PtrStepSz<int> depthMat,          
                                   unsigned int width,
                                   unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    // check to make sure center pixel is valid, and there is range for the 3x3 window
    if ( (iCol >= 1)           &&
         (iCol < (width - 1))  &&
         (iRow >= 1)           &&
         (iRow < (height - 1)) &&
         (maskMatPtr(iRow, iCol) == 1)) {
    
        // this implements a filter with potentially missing data, as specified by cMatPtr.  
        // disregard the entire row if either end pixel is missing
        // skip the center column because the coefficients are zero
        int rowMasks[3] = {
        maskMatPtr(iRow-1, iCol-1)*maskMatPtr(iRow-1, iCol+1), 
        maskMatPtr(iRow,   iCol-1)*maskMatPtr(iRow,   iCol+1), 
        maskMatPtr(iRow+1, iCol-1)*maskMatPtr(iRow+1, iCol+1)  };

        // this is the convolution of the gx kernel over the 3x3 pixel window
        // skip the center column because the coefficients are zero
        int sumgx = 
        gx[0]*rowMasks[0]*depthMat(iRow-1,iCol-1) + gx[2]*rowMasks[0]*depthMat(iRow-1,iCol+1) +
        gx[3]*rowMasks[1]*depthMat(iRow,  iCol-1) + gx[5]*rowMasks[1]*depthMat(iRow,  iCol+1) +
        gx[6]*rowMasks[2]*depthMat(iRow+1,iCol-1) + gx[8]*rowMasks[2]*depthMat(iRow+1,iCol+1);

        // disregard entire column if either end pixel is missing
        int colMasks[3] = {
            maskMatPtr(iRow-1, iCol-1)*maskMatPtr(iRow+1, iCol-1),  
            maskMatPtr(iRow-1, iCol  )*maskMatPtr(iRow+1, iCol  ),  
            maskMatPtr(iRow-1, iCol+1)*maskMatPtr(iRow+1, iCol+1)  };

        // this is the convolution of the gy kernel over the 3x3 pixel window
        // skip the center column because the coefficients are zero
        int sumgy = 
        gy[0]*colMasks[0]*depthMat(iRow-1,iCol-1) + gy[1]*colMasks[1]*depthMat(iRow-1,iCol) + gy[2]*colMasks[2]*depthMat(iRow-1,iCol+1) +
        gy[6]*colMasks[0]*depthMat(iRow+1,iCol-1) + gy[7]*colMasks[1]*depthMat(iRow+1,iCol) + gy[8]*colMasks[2]*depthMat(iRow+1,iCol+1);    

        float sumgx_f = __int2float_rn(sumgx);
        float sumgy_f = __int2float_rn(sumgy);
        float gradNorm = gradScale * sqrt(sumgx_f*sumgx_f + sumgy_f*sumgy_f);

        if (gradNorm > 0.5) {
           edgeMaskMat(iRow, iCol) = 0; 

           // also set the converged flag for the pixel to 0
           // maskMatPtr(iRow, iCol) = 0;  // but don't do this because it will affect the other pixels in this loop
        }
    } else {
        edgeMaskMat(iRow, iCol) = 0;
    }
}


//-----------------------------------------------------------------------------
// this function smoothes 3d points in the point cloud by averaging the point location
// with it's adjacement neighbors
__global__ void pointSmoothingKernel(cv::cuda::PtrStepSz<float3> posMatSmoothedPtr_m,  // outputs
                                     cv::cuda::PtrStepSz<float3> posMatPtr_m,          // inputs
                                     cv::cuda::PtrStepSz<int>    cMatPtr,
                                     cv::cuda::PtrStepSz<float>  pixelXPosMat,
                                     cv::cuda::PtrStepSz<float>  pixelYPosMat,
                                     unsigned int width,
                                     unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    int l = 1;

    if ( (iCol >= l)         &&
         (iCol < (width-l))  &&
         (iRow >= l)         &&
         (iRow < (height-l)) &&
         (cMatPtr(iRow, iCol) == 1) ) 
    {

        // 9 points for the grid
        float3 r0 = posMatPtr_m(iRow,   iCol)   * __int2float_rn(cMatPtr(iRow,   iCol));  // center pixel
        float3 r1 = posMatPtr_m(iRow+l, iCol)   * __int2float_rn(cMatPtr(iRow+l, iCol));
        float3 r2 = posMatPtr_m(iRow+l, iCol+l) * __int2float_rn(cMatPtr(iRow+l, iCol+l));
        float3 r3 = posMatPtr_m(iRow,   iCol+l) * __int2float_rn(cMatPtr(iRow,   iCol+l));
        float3 r4 = posMatPtr_m(iRow-l, iCol+l) * __int2float_rn(cMatPtr(iRow-l, iCol+l));
        float3 r5 = posMatPtr_m(iRow-l, iCol)   * __int2float_rn(cMatPtr(iRow-l, iCol));
        float3 r6 = posMatPtr_m(iRow-l, iCol-l) * __int2float_rn(cMatPtr(iRow-l, iCol-l));
        float3 r7 = posMatPtr_m(iRow,   iCol-l) * __int2float_rn(cMatPtr(iRow,   iCol-l));
        float3 r8 = posMatPtr_m(iRow+l, iCol-l) * __int2float_rn(cMatPtr(iRow+l, iCol-l));

        int sumi = cMatPtr(iRow,   iCol)   + 
                   cMatPtr(iRow+l, iCol)   + 
                   cMatPtr(iRow+l, iCol+l) + 
                   cMatPtr(iRow,   iCol+l) + 
                   cMatPtr(iRow-l, iCol+l) +
                   cMatPtr(iRow-l, iCol)   + 
                   cMatPtr(iRow-l, iCol-l) + 
                   cMatPtr(iRow,   iCol-l) + 
                   cMatPtr(iRow+l, iCol-l);

        float3 rMean = (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8) / __int2float_rn(sumi);

        // shift center point some ratio closer to the mean
        posMatSmoothedPtr_m(iRow, iCol) = (rMean * k_ave) + r0 * (1-k_ave);

        //posMatSmoothedPtr_m(iRow,   iCol) = rMean;
    }
}


//-----------------------------------------------------------------------------
// computes the normal vector (perpendicular to a surface) of each valid depth pixel
__global__ void computeNormalKernel(cv::cuda::PtrStepSz<float3> normMatPtr,   // outputs
                                    cv::cuda::PtrStepSz<float3> posMatPtr_m,  // inputs
                                    cv::cuda::PtrStepSz<int>    cMatPtr,
                                    cv::cuda::PtrStepSz<float> pixelXPosMat,
                                    cv::cuda::PtrStepSz<float> pixelYPosMat,
                                    unsigned int width,
                                    unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    if ( (iCol > 0)          &&
         (iCol < (width-1))  &&
         (iRow > 0)          &&
         (iRow < (height-1)) &&
         (cMatPtr(iRow, iCol) == 1) ) 

    {
        // 9 points for the grid
        float3 r0 = posMatPtr_m(iRow, iCol)            * __int2float_rn(cMatPtr(iRow,   iCol));
        float3 r1 = (posMatPtr_m(iRow+1, iCol)   - r0) * __int2float_rn(cMatPtr(iRow+1, iCol));
        float3 r2 = (posMatPtr_m(iRow+1, iCol+1) - r0) * __int2float_rn(cMatPtr(iRow+1, iCol+1));
        float3 r3 = (posMatPtr_m(iRow,   iCol+1) - r0) * __int2float_rn(cMatPtr(iRow,   iCol+1));
        float3 r4 = (posMatPtr_m(iRow-1, iCol+1) - r0) * __int2float_rn(cMatPtr(iRow-1, iCol+1));
        float3 r5 = (posMatPtr_m(iRow-1, iCol)   - r0) * __int2float_rn(cMatPtr(iRow-1, iCol));
        float3 r6 = (posMatPtr_m(iRow-1, iCol-1) - r0) * __int2float_rn(cMatPtr(iRow-1, iCol-1));
        float3 r7 = (posMatPtr_m(iRow,   iCol-1) - r0) * __int2float_rn(cMatPtr(iRow,   iCol-1));
        float3 r8 = (posMatPtr_m(iRow+1, iCol-1) - r0) * __int2float_rn(cMatPtr(iRow+1, iCol-1));

        r1 = normalize(r1);
        r2 = normalize(r2);
        r3 = normalize(r3);
        r4 = normalize(r4);
        r5 = normalize(r5);
        r6 = normalize(r6);
        r7 = normalize(r7);
        r8 = normalize(r8);

        float a11 = r1.x*r1.x + r2.x*r2.x + r3.x*r3.x + r4.x*r4.x + 
                    r5.x*r5.x + r6.x*r6.x + r7.x*r7.x + r8.x*r8.x;

        float a22 = r1.y*r1.y + r2.y*r2.y + r3.y*r3.y + r4.y*r4.y + 
                    r5.y*r5.y + r6.y*r6.y + r7.y*r7.y + r8.y*r8.y;

        float a33 = r1.z*r1.z + r2.z*r2.z + r3.z*r3.z + r4.z*r4.z + 
                    r5.z*r5.z + r6.z*r6.z + r7.z*r7.z + r8.z*r8.z;

        float a12 = r1.x*r1.y + r2.x*r2.y + r3.x*r3.y + r4.x*r4.y + 
                    r5.x*r5.y + r6.x*r6.y + r7.x*r7.y + r8.x*r8.y;

        float a13 = r1.x*r1.z + r2.x*r2.z + r3.x*r3.z + r4.x*r4.z + 
                    r5.x*r5.z + r6.x*r6.z + r7.x*r7.z + r8.x*r8.z;

        float a23 = r1.y*r1.z + r2.y*r2.z + r3.y*r3.z + r4.y*r4.z + 
                    r5.y*r5.z + r6.y*r6.z + r7.y*r7.z + r8.y*r8.z;     

        float U[9];
        float S[9];
        float V[9];

        svd(a11,  a12,  a13,  a12,  a22,  a23,  a13,  a23,  a33,
            U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
            S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8],
            V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);  

        float3 normUvec = normalize(make_float3(V[6], V[7], V[8])) * sign(V[8]);

        // check to make sure normal isn't perpendicular to camera line of sight
        // float3 pUvec = normalize(make_float3(pixelXPosMat(iRow, iCol), pixelYPosMat(iRow, iCol), 1.0));

        // if (abs(dot(pUvec, normUvec)) < angleThresh) {
        // //if (normUvec.z < angleThresh) {            
        //     normUvec = normUvec * 0.0;
        // }
        normMatPtr(iRow, iCol) = normUvec;
    }
}


//-----------------------------------------------------------------------------
// Kalman filter function.  
// - states are stored externally, passed in each time
// - propagation and correction done in one function here
__global__ void kalmanFilterKernel( cv::cuda::PtrStepSz<int> convergedMatPtr,    // outputs
                                    cv::cuda::PtrStepSz<int> depthMatEstPtr_mm,  // states
                                    cv::cuda::PtrStepSz<float> depthMatVarPtr_mm2,
                                    cv::cuda::PtrStepSz<int> depthMatPtr_mm,     // inputs
                                    unsigned int width,
                                    unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;  

    // no state propagation right now
    // x_k+1 = x_k

    // propagate variance
    depthMatVarPtr_mm2(iRow, iCol) = depthMatVarPtr_mm2(iRow, iCol) + processNoiseVar_mm2;

    // correct
    float measurement_mm = __int2float_rn(depthMatPtr_mm(iRow, iCol));

    if ((measurement_mm > 100.0) && (measurement_mm < 60000.0)){
        convergedMatPtr(iRow, iCol) = 1;

        float estimate_mm = __int2float_rn(depthMatEstPtr_mm(iRow, iCol));
        float residual_mm = measurement_mm - estimate_mm;

        // compute kalman gain
        float S = depthMatVarPtr_mm2(iRow, iCol) + measNoiseVar_mm2;
        float K = depthMatVarPtr_mm2(iRow, iCol) / S;

        estimate_mm = estimate_mm + K * residual_mm;

        depthMatEstPtr_mm(iRow, iCol) = __float2int_rn(estimate_mm);

        depthMatVarPtr_mm2(iRow, iCol) = (1.0f - K) * depthMatVarPtr_mm2(iRow, iCol);
    }

    // limit variance
    if (depthMatVarPtr_mm2(iRow, iCol) >= depthVarMax_mm2) {
        depthMatVarPtr_mm2(iRow, iCol) = depthVarMax_mm2;
    } else if (depthMatVarPtr_mm2(iRow, iCol) < depthVarMin_mm2) {
        depthMatVarPtr_mm2(iRow, iCol) = depthVarMin_mm2;
    }

    // check to see if the position is known well enough
    // if (depthMatVarPtr_mm2(iRow, iCol) < depthConvergedThresh_mm2){     
    //    convergedMatPtr(iRow, iCol) = 1;
    //}
}


//-----------------------------------------------------------------------------
__global__ void depthToPositionKernel(cv::cuda::PtrStepSz<float3> posMatPtr_m,   // ouputs           
                                      cv::cuda::PtrStepSz<int> depthMatPtr_mm,   // inputs
                                      cv::cuda::PtrStepSz<float> pixelXPosMatPtr,  
                                      cv::cuda::PtrStepSz<float> pixelYPosMatPtr,
                                      unsigned int width,
                                      unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    float z_m = __int2float_rn(depthMatPtr_mm(iRow, iCol)) * 0.001;
    float x_m = pixelXPosMatPtr(iRow, iCol) * z_m;
    float y_m = pixelYPosMatPtr(iRow, iCol) * z_m;

    posMatPtr_m(iRow, iCol) = make_float3(x_m, y_m, z_m);
}


//-----------------------------------------------------------------------------
// applies a rotation and transformation to the 3-d point cloud data, then re-computes
// the expected depth image, taking into account any shading that may have occured due
// to the perspective change
__global__ void transformPositionToDepthKernel(cv::cuda::PtrStepSz<int> depthMatRotatedPtr_mm,  // outputs
                                            cv::cuda::PtrStepSz<int> depthShadedPtr_mm,
                                            cv::cuda::PtrStepSz<int> shadingMaskMatPtr,
                                            cv::cuda::PtrStepSz<int> xIdNewMatPtr,
                                            cv::cuda::PtrStepSz<int> yIdNewMatPtr,
                                            cv::cuda::PtrStepSz<float3> posMatPtr_m,            // inputs          
                                            unsigned int width,
                                            unsigned int height,
                                            float *dcmDepthtoColor,
                                            float *posDepthToColor_m,
                                            bool useRgbCoeff)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    uint2 pixelXY = make_uint2(0, 0);
    float3 pointFrame1_m = posMatPtr_m(iRow, iCol);
    float3 pointFrame2_m = make_float3(0, 0, 0);

    bool pixelInRange = false;
    bool depthInRange = true;

    if (pointFrame1_m.z < 0.1f) {
        depthInRange = false;
    }

    pointFrame2_m = rotateTranslate(dcmDepthtoColor, posDepthToColor_m, pointFrame1_m);

    if (useRgbCoeff){
        pixelInRange = projectPointToPixel(pixelXY,        // outputs
                                           pointFrame2_m,  // inputs
                                           rgbPixelCoeffs,
                                           rgbLensCoeffs,
                                           width,
                                           height);
    } else {
        pixelInRange = projectPointToPixel(pixelXY,         // outputs
                                           pointFrame2_m,   // inputs
                                           depthPixelCoeffs,
                                           depthLensCoeffs,
                                           width,
                                           height); 
    }

    if (depthInRange && pixelInRange){

        int depthZ_mm = __float2int_rn(pointFrame2_m.z * 1000.0f);
        depthMatRotatedPtr_mm(iRow, iCol) = depthZ_mm;

        // this is key to depth testing. due to the transformation there
        // can be many points that have the same pixel x,y coordinates.
        // only replace the coordinates with a new value if it is closer
        // to the camera (technically it's comparing the z coordinate in linear
        // space, but this works because it should only be comparing to other points
        // oocupying that pixel location).
        atomicMin(&depthShadedPtr_mm(pixelXY.y, pixelXY.x), depthZ_mm);

        shadingMaskMatPtr(iRow, iCol) = 1;

        xIdNewMatPtr(iRow, iCol) = pixelXY.x;
        yIdNewMatPtr(iRow, iCol) = pixelXY.y;
    }
}


//-----------------------------------------------------------------------------
// computes R,G,B of the depth points.  the depth and rgb cameras are at different
// locations in the sensor, so this function corrects for that.  
__global__ void generateShadedColorImageKernel( cv::cuda::PtrStepSz<int> shadedColorMatPtr_r, // outputs
                                                cv::cuda::PtrStepSz<int> shadedColorMatPtr_g,
                                                cv::cuda::PtrStepSz<int> shadedColorMatPtr_b,
                                                cv::cuda::PtrStepSz<int> clrShadedMaskMatPtr,
                                                cv::cuda::PtrStepSz<uchar3> colorMatPtr,      // inputs
                                                cv::cuda::PtrStepSz<int> depthMatRotatedPtr_mm,   
                                                cv::cuda::PtrStepSz<int> depthShadedPtr_mm,
                                                cv::cuda::PtrStepSz<int> depthShadedMaskMatPtr,
                                                cv::cuda::PtrStepSz<int> xIdNewMatPtr,
                                                cv::cuda::PtrStepSz<int> yIdNewMatPtr,
                                                unsigned int width,
                                                unsigned int height)
{
    
    // inputs
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    if (depthShadedMaskMatPtr(iRow, iCol) == 1) {

        int mappedRow = yIdNewMatPtr(iRow, iCol);
        int mappedCol = xIdNewMatPtr(iRow, iCol);

        // check to see if this pixel is the closest one, using the shading computed by the 
        // transformDepthImageKernel function
        if (depthMatRotatedPtr_mm(iRow, iCol) <= depthShadedPtr_mm(mappedRow, mappedCol)) {

            // assume the transformation represents the depth to camera frames,
            // so look up the color in the newly computed color frame, and assign it to the
            // depth frame color
            shadedColorMatPtr_r(iRow, iCol) = int(colorMatPtr(mappedRow, mappedCol).z);
            shadedColorMatPtr_g(iRow, iCol) = int(colorMatPtr(mappedRow, mappedCol).y);
            shadedColorMatPtr_b(iRow, iCol) = int(colorMatPtr(mappedRow, mappedCol).x);

            clrShadedMaskMatPtr(iRow, iCol) = 1;
        }
    }
}


//-----------------------------------------------------------------------------
// combines the three R,G,B channels into one image  
__global__ void combineRGB(cv::cuda::PtrStepSz<uchar3> colorInDepthMatPtr,   // outputs
                           cv::cuda::PtrStepSz<int> colorInDepthMatPtr_r,    // inputs
                           cv::cuda::PtrStepSz<int> colorInDepthMatPtr_g,
                           cv::cuda::PtrStepSz<int> colorInDepthMatPtr_b)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    colorInDepthMatPtr(iRow, iCol) = make_uchar3(char(colorInDepthMatPtr_b(iRow, iCol)), 
                                                 char(colorInDepthMatPtr_g(iRow, iCol)),
                                                 char(colorInDepthMatPtr_r(iRow, iCol)) );
}    


//-----------------------------------------------------------------------------
// given a new pixel mapping due to a perspective change, re-compute all the associated
// images with this new pixel mapping
__global__ void transformMatsKernel( cv::cuda::PtrStepSz<float> depthMatVarTransformedPtr_mm2, // outputs
                                     cv::cuda::PtrStepSz<uchar3> colorMatTransformedPtr, 
                                     cv::cuda::PtrStepSz<float> depthMatVarPtr_mm2,            // inputs
                                     cv::cuda::PtrStepSz<uchar3> colorMatPtr,        
                                     cv::cuda::PtrStepSz<int> depthMatRotatedPtr_mm,   
                                     cv::cuda::PtrStepSz<int> depthShadedPtr_mm,
                                     cv::cuda::PtrStepSz<uchar> MaskMatPtr,
                                     cv::cuda::PtrStepSz<int> xIdNewMatPtr,
                                     cv::cuda::PtrStepSz<int> yIdNewMatPtr,
                                     unsigned int width,
                                     unsigned int height)
{
    
    // inputs
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    // if no data is available for the pixel, make it magenta
    uchar3 color = make_uchar3(160, 0, 160);

    if (MaskMatPtr(iRow, iCol) == 1) {

        int mappedRow = yIdNewMatPtr(iRow, iCol);
        int mappedCol = xIdNewMatPtr(iRow, iCol);

        // check to see if this pixel is the closest one, using the shading computed by the 
        // transformDepthImageKernel function
        if (depthMatRotatedPtr_mm(iRow, iCol) <= depthShadedPtr_mm(mappedRow, mappedCol)) {

            // assume the depth and color images are maped to the same depth frame, so rotate both 
            // from index (iRow, iCol) in the original input frame to (mappedRow, MappedCol) in
            // the transformed frame.
            depthMatVarTransformedPtr_mm2(mappedRow, mappedCol) = depthMatVarPtr_mm2(iRow, iCol);
            colorMatTransformedPtr(mappedRow, mappedCol) = colorMatPtr(iRow, iCol);
        }
    }
}


//-----------------------------------------------------------------------------
// assigns 3-d points and color to pointer array that is used by OpenGL for visualization
__global__ void mapToOpenGLKernel(float3 *pointsOut,                          // outputs
                                  uchar3 *colorsOut,  
                                  cv::cuda::PtrStepSz<uchar> depthMat8L_mm,
                                  cv::cuda::PtrStepSz<uchar> depthMat8U_mm,                            
                                  cv::cuda::PtrStepSz<int> convergedMaskMat,  // inputs
                                  cv::cuda::PtrStepSz<float3> posMatPtr_m,
                                  cv::cuda::PtrStepSz<int> depthMatPtr_mm,
                                  cv::cuda::PtrStepSz<uchar3> colorInDepthMat,
                                  unsigned int width,
                                  unsigned int height)
{
    
    // inputs
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    float3 point_m  = make_float3(0, 0, 0);  
    float tmp1 = 0; 
    int depthL = 0;
    int depthU = 0;

    // check to make sure variance of pixel is below a threshold
    if (convergedMaskMat(iRow, iCol) == 1) {

        float3 tmp = posMatPtr_m(iRow, iCol);
        point_m = make_float3(tmp.x, -tmp.y, -tmp.z); 

        tmp1 = __int2float_rn(depthMatPtr_mm(iRow, iCol));
        depthL = __int2float_rd(tmp1 / 255.0f);
        depthU = depthMatPtr_mm(iRow, iCol) - (depthL * 255);
    }

    // openGL and openCV have different RGB orders
    pointsOut[iRow * width + iCol] = point_m;
    colorsOut[iRow * width + iCol] = colorInDepthMat(iRow, iCol);

    depthMat8L_mm(iRow, iCol) = char(depthL);
    depthMat8U_mm(iRow, iCol) = char(depthU);
} 


///////////////////////////////////////////////////////////////////////////////
// kernel wrapper functions, called in .cpp files
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
extern "C"
void runDepthProcessing(cv::cuda::GpuMat posMat_m,             // outputs
                       cv::cuda::GpuMat convergedMaskMat,
                       cv::cuda::GpuMat depthMatBlurred,
                       cv::cuda::GpuMat edgeMaskMat,
                       cv::cuda::GpuMat posMatSmoothed_m,
                       cv::cuda::GpuMat normMat,
                       cv::cuda::GpuMat depthMatEst_mm,       // states
                       cv::cuda::GpuMat depthMatVar_mm2,
                       const cv::cuda::GpuMat depthMat_mm,    // inputs
                       const cv::cuda::GpuMat pixelXPosMat,
                       const cv::cuda::GpuMat pixelYPosMat,
                       const unsigned int width,
                       const unsigned int height)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    kalmanFilterKernel<<<grid, block>>>(convergedMaskMat, // outputs
                                        depthMatEst_mm,  // states
                                        depthMatVar_mm2,
                                        depthMat_mm,     // inputs
                                        width,
                                        height); 

    GaussianBlur5x5Kernel<<<grid, block>>>(depthMatBlurred, // outputs
                                           depthMat_mm,     // inputs
                                           convergedMaskMat,
                                           width,
                                           height);

    edgeDetectorKernel<<<grid, block>>>(edgeMaskMat,       // outputs
                                        convergedMaskMat,  // inputs
                                        depthMatBlurred, 
                                        width,
                                        height);

    GaussianBlur5x5Kernel<<<grid, block>>>(depthMatBlurred, // outputs
                                           depthMatEst_mm,  // inputs
                                           edgeMaskMat,
                                           width,
                                           height);



    // depthMatBlurred.setTo(0);
    // blur again, this time ignoring the edge points when computing the convolution
    // GaussianBlur5x5Kernel<<<grid, block>>>(depthMatBlurred,  // outputs
    //                                        depthMatEst_mm,   // inputs
    //                                        convergedMakMat,
    //                                        width,
    //                                        height);

    // depthMatEst_mm = depthMatBlurred.clone();

    // depthToPositionKernel<<<grid, block>>>(posMat_m,       // ouputs           
    //                                        depthMatEst_mm, // inputs
    //                                        pixelXPosMat,  
    //                                        pixelYPosMat,
    //                                        width,
    //                                        height); 

    // for (unsigned int iter = 0; iter < 1; iter++) {
    //     pointSmoothingKernel<<<grid, block>>>(  posMatSmoothed_m,  // outputs
    //                                         posMat_m,              // inputs
    //                                         convergedMat,
    //                                         pixelXPosMat,
    //                                         pixelYPosMat,
    //                                         width,
    //                                         height);
    //     posMat_m = posMatSmoothed_m.clone();
    // }
    // posMatSmoothed_m = posMat_m.clone();

    // computeNormalKernel<<<grid, block>>>(normMat,       // outputs
    //                                     posMat_m,       // inputs
    //                                     convergedMaskMat,
    //                                     pixelXPosMat,
    //                                     pixelYPosMat,
    //                                     width,
    //                                     height);                                                                                 
}


//-----------------------------------------------------------------------------
// the RGB camera is at a different location and orientation from the depth reciever
// compute an RGB image with pixels 1:1 with the depth image (as best as possible using given calibration coefficients)
extern "C"
void syncColorToDepthData( cv::cuda::GpuMat colorInDepthMat_r,      // outputs
                           cv::cuda::GpuMat colorInDepthMat_g,
                           cv::cuda::GpuMat colorInDepthMat_b,
                           cv::cuda::GpuMat depthMatRotated_mm,
                           cv::cuda::GpuMat depthMatShaded_mm,
                           cv::cuda::GpuMat depthShadedMaskMat,
                           cv::cuda::GpuMat clrShadedMaskMat,
                           cv::cuda::GpuMat xIdNewMat,
                           cv::cuda::GpuMat yIdNewMat,
                           cv::cuda::GpuMat posMat_m,
                           const cv::cuda::GpuMat depthMat_mm,   // inputs
                           const cv::cuda::GpuMat colorMat,
                           const cv::cuda::GpuMat pixelXPosMat,
                           const cv::cuda::GpuMat pixelYPosMat,
                           const unsigned int width,
                           const unsigned int height,
                           float *dcmDepthtoColor,
                           float *posDepthToColor_m)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    bool useRgbCoeffs = true;

    // for going from the 3D points to the RGB pixel locations, use the rgb lense coeffs
    depthToPositionKernel<<<grid, block>>>(posMat_m,   // ouputs           
                                          depthMat_mm,  // inputs
                                          pixelXPosMat,  
                                          pixelYPosMat,
                                          width,
                                          height);

    transformPositionToDepthKernel<<<grid, block>>>(depthMatRotated_mm, // outputs
                                                    depthMatShaded_mm,
                                                    depthShadedMaskMat,
                                                    xIdNewMat,
                                                    yIdNewMat,
                                                    posMat_m,           // inputs          
                                                    width,
                                                    height,
                                                    dcmDepthtoColor,
                                                    posDepthToColor_m,
                                                    useRgbCoeffs);

    // in this function, map the raw colorMat to the depth frame and store it in colorInDepthMat
    generateShadedColorImageKernel<<<grid, block>>>(colorInDepthMat_r,   // outputs
                                                    colorInDepthMat_g,    
                                                    colorInDepthMat_b,   
                                                    clrShadedMaskMat,
                                                    colorMat,            // inputs
                                                    depthMatRotated_mm,   
                                                    depthMatShaded_mm,
                                                    depthShadedMaskMat,
                                                    xIdNewMat,
                                                    yIdNewMat,
                                                    width,
                                                    height);  

}


//-----------------------------------------------------------------------------
// the RGB camera is at a different location and orientation from the depth reciever
// compute an RGB image with pixels 1:1 with the depth image (as best as possible using given calibration coefficients)
extern "C"
void fillMissingColorData(cv::cuda::GpuMat colorInDepthMat,       // outputs 
                          cv::cuda::GpuMat clrInDepthMatBlur_r,   // working variables / states
                          cv::cuda::GpuMat clrInDepthMatBlur_g,
                          cv::cuda::GpuMat clrInDepthMatBlur_b,
                          cv::cuda::GpuMat clrShadedMaskBlurMat,
                          cv::cuda::GpuMat clrInDepthMat_r,       // inputs
                          cv::cuda::GpuMat clrInDepthMat_g,    
                          cv::cuda::GpuMat clrInDepthMat_b,
                          cv::cuda::GpuMat clrShadedMaskMat,     
                          const unsigned int width,
                          const unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    int thresh;
    thresh = 17;
    fillInColorKernel<<<grid, block>>>( clrInDepthMatBlur_r,  // outputs
                                        clrInDepthMatBlur_g,
                                        clrInDepthMatBlur_b,
                                        clrShadedMaskBlurMat,
                                        clrShadedMaskMat,     // inputs
                                        clrInDepthMat_r,      
                                        clrInDepthMat_g,
                                        clrInDepthMat_b,
                                        thresh,
                                        width,
                                        height);

     combineRGB<<<grid, block>>>(colorInDepthMat,        // outputs
                                 clrInDepthMatBlur_r,    // inputs
                                 clrInDepthMatBlur_g,
                                 clrInDepthMatBlur_b);                                                                                                                                                                                    
}


//-----------------------------------------------------------------------------
extern "C"
void transformData(cv::cuda::GpuMat colorInDepthMatTransformed,   // outputs
                   cv::cuda::GpuMat depthMatVarTransformed_mm2,
                   cv::cuda::GpuMat depthMatTransformed_mm,
                   cv::cuda::GpuMat depthMatRotated_mm,
                   cv::cuda::GpuMat shadingMaskMat,
                   cv::cuda::GpuMat xIdNewMat,
                   cv::cuda::GpuMat yIdNewMat,
                   cv::cuda::GpuMat posMat_m,    
                   const cv::cuda::GpuMat depthMat_mm,            // inputs 
                   const cv::cuda::GpuMat depthMatVar_mm2,         
                   const cv::cuda::GpuMat colorMatInput,  
                   const cv::cuda::GpuMat depthMatMeas_mm,                     
                   const cv::cuda::GpuMat pixelXPosMat,
                   const cv::cuda::GpuMat pixelYPosMat,
                   const unsigned int width,
                   const unsigned int height,
                   float *dcmRotation,
                   float *posTrans_m)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    bool useRgbCoeffs = false;

    // depth must be transformed first.  since depth contains the z, information,
    // and with the pixel locations contains the (x, y, z) information for each point
    // so the depth image is all that is necessary to compute the transformed + shaded
    // depth image, as well as pixel location indexes for the transformed variance and
    // color images.
    depthToPositionKernel<<<grid, block>>>(posMat_m,     // ouputs           
                                           depthMat_mm,  // inputs
                                           pixelXPosMat,  
                                           pixelYPosMat,
                                           width,
                                           height);  

    transformPositionToDepthKernel<<<grid, block>>>(depthMatRotated_mm, // outputs
                                                    depthMatTransformed_mm,
                                                    shadingMaskMat,
                                                    xIdNewMat,
                                                    yIdNewMat,
                                                    posMat_m,           // inputs           
                                                    width,
                                                    height,
                                                    dcmRotation,
                                                    posTrans_m,
                                                    useRgbCoeffs);

    transformMatsKernel<<<grid, block>>>( depthMatVarTransformed_mm2, // outputs
                                          colorInDepthMatTransformed,  
                                          depthMatVar_mm2,            // inputs
                                          colorMatInput,               
                                          depthMatRotated_mm,   
                                          depthMatTransformed_mm,
                                          shadingMaskMat,
                                          xIdNewMat,
                                          yIdNewMat,
                                          width,
                                          height);  
}


//-----------------------------------------------------------------------------
// the RGB camera is at a different location and orientation from the depth reciever
// compute an RGB image with pixels 1:1 with the depth image (as best as possible using given calibration coefficients)
extern "C"
void mapColorDepthToOpenGL(float3 *pointsOut,                        // outputs
                           uchar3 *colorsOut, 
                           const cv::cuda::GpuMat depthMat8L_mm,
                           const cv::cuda::GpuMat depthMat8U_mm,
                           const cv::cuda::GpuMat convergedMaskMat,  // inputs
                           const cv::cuda::GpuMat posMat_m,       
                           const cv::cuda::GpuMat depthMat_mm,
                           const cv::cuda::GpuMat colorInDepthMat,
                           const unsigned int width,
                           const unsigned int height)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // map the vertex 3D points and color mat to openGL VBOs in the main.cpp
    mapToOpenGLKernel<<<grid, block>>>(pointsOut,         // outputs
                                       colorsOut,  
                                       depthMat8L_mm,
                                       depthMat8U_mm,                             
                                       convergedMaskMat,  // inputs
                                       posMat_m,
                                       depthMat_mm,
                                       colorInDepthMat,
                                       width,
                                       height);   
}


