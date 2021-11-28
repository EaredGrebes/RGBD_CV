#include <math_constants.h>
#include <opencv2/cudaarithm.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>

//#include "cuda_computer_vision_kernel.cu"
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

//                                      ppx      ppx      fx       fy
__constant__ float depthPixelCoeffs[4] = {423.337, 238.688, 421.225, 421.225};

//  should be zero, see this:
// https://github.com/IntelRealSense/librealsense/issues/1430
__constant__ float depthLensCoeffs[5] = {0, 0, 0, 0, 0};

// kalman filter parameters 
__constant__ float processNoiseVar_mm2 = 15.0f;
__constant__ float measNoiseVar_mm2 = 2000.0f;

__constant__ float depthVarMax_mm2 = 5000 * 5000;
__constant__ float depthVarMin_mm2 = 4;


__constant__ float depthConvergedThresh_mm2 = 900.0f;

// this coresponds to throwing out measurements where the camera line of sight 
// for a given pixel is 80 deg off from the estimated normal of that pixel
// cos(80 deg) = 0.17
__constant__ float angleThresh = 0.17f;

__constant__ float k_ave = 0.3f;   

// the gaussian kernel for image blurring
__constant__ int gK[25] = {1, 4,  7,  4,  1,
                           4, 16, 26, 16, 4,
                           7, 26, 41, 26, 7,
                           4, 16, 26, 16, 4,
                           1, 4,  7,  4,  1};

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


///////////////////////////////////////////////////////////////////////////////
// Kernel functions
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
__global__ void GaussianBlur5x5Kernel(cv::cuda::PtrStepSz<int> depthMatBlurred, // outputs
                                      cv::cuda::PtrStepSz<int> depthMat,        // inputs
                                      cv::cuda::PtrStepSz<int> cMatPtr,
                                      unsigned int width,
                                      unsigned int height)
{
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y; 

    if ( (iCol >= 2)           &&
         (iCol < (width - 2))  &&
         (iRow >= 2)           &&
         (iRow < (height - 2)) &&
         (cMatPtr(iRow, iCol) == 1)) {

        // this implements a filter with potentially missing data, as specified by cMatPtr.  
        int gKm[25] = {
        gK[0] *cMatPtr(iRow-2,iCol-2), gK[1] *cMatPtr(iRow-2,iCol-1), gK[2] *cMatPtr(iRow-2,iCol), gK[3] *cMatPtr(iRow-2,iCol+1), gK[4] *cMatPtr(iRow-2, iCol+2), 
        gK[5] *cMatPtr(iRow-1,iCol-2), gK[6] *cMatPtr(iRow-1,iCol-1), gK[7] *cMatPtr(iRow-1,iCol), gK[8] *cMatPtr(iRow-1,iCol+1), gK[9] *cMatPtr(iRow-1, iCol+1), 
        gK[10]*cMatPtr(iRow,  iCol-2), gK[11]*cMatPtr(iRow,  iCol-1), gK[12]*cMatPtr(iRow,  iCol), gK[13]*cMatPtr(iRow,  iCol+1), gK[14]*cMatPtr(iRow,  iCol+2), 
        gK[15]*cMatPtr(iRow+1,iCol-2), gK[16]*cMatPtr(iRow+1,iCol-1), gK[17]*cMatPtr(iRow+1,iCol), gK[18]*cMatPtr(iRow+1,iCol+1), gK[19]*cMatPtr(iRow+1,iCol+2),
        gK[20]*cMatPtr(iRow+2,iCol-2), gK[21]*cMatPtr(iRow+2,iCol-1), gK[22]*cMatPtr(iRow+2,iCol), gK[23]*cMatPtr(iRow+2,iCol+1), gK[24]*cMatPtr(iRow+2,iCol+2) };

        // this is the convolution of the gaussian kernel over the 5x5 pixel window
        int sum = 
        gKm[0] *depthMat(iRow-2,iCol-2) + gKm[1] *depthMat(iRow-2,iCol-1) + gKm[2] *depthMat(iRow-2,iCol) + gKm[3] *depthMat(iRow-2,iCol+1) + gKm[4] *depthMat(iRow-2, iCol+2) +
        gKm[5] *depthMat(iRow-1,iCol-2) + gKm[6] *depthMat(iRow-1,iCol-1) + gKm[7] *depthMat(iRow-1,iCol) + gKm[8] *depthMat(iRow-1,iCol+1) + gKm[9] *depthMat(iRow-1, iCol+1) +
        gKm[10]*depthMat(iRow,  iCol-2) + gKm[11]*depthMat(iRow,  iCol-1) + gKm[12]*depthMat(iRow,  iCol) + gKm[13]*depthMat(iRow,  iCol+1) + gKm[14]*depthMat(iRow,  iCol+2) +
        gKm[15]*depthMat(iRow+1,iCol-2) + gKm[16]*depthMat(iRow+1,iCol-1) + gKm[17]*depthMat(iRow+1,iCol) + gKm[18]*depthMat(iRow+1,iCol+1) + gKm[19]*depthMat(iRow+1,iCol+2) +
        gKm[20]*depthMat(iRow+2,iCol-2) + gKm[21]*depthMat(iRow+2,iCol-1) + gKm[22]*depthMat(iRow+2,iCol) + gKm[23]*depthMat(iRow+2,iCol+1) + gKm[24]*depthMat(iRow+2,iCol+2);
    
        // compute the sum of the filter coefficients
        int sum2 = gKm[0]  + gKm[1]  + gKm[2]  + gKm[3] + gKm[4] + gKm[5] + gKm[6] + gKm[7] + gKm[8] + gKm[9] + gKm[10] + gKm[11] + gKm[12] + gKm[13] + gKm[14] + gKm[15] 
                 + gKm[16] + gKm[17] + gKm[18] + gKm[19]+ gKm[20]+ gKm[21]+ gKm[22]+ gKm[23]+ gKm[24];

        int tmp3 = __float2int_rn(__int2float_rn(sum) / __int2float_rn(sum2));
    
        depthMatBlurred(iRow, iCol) = tmp3;
    }
}


//-----------------------------------------------------------------------------
__global__ void edgeDetectorKernel(cv::cuda::PtrStepSz<float> depthMatEdges,   // outputs
                                   cv::cuda::PtrStepSz<int> cMatPtr,
                                   cv::cuda::PtrStepSz<int> depthMat,          // inputs
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
         (cMatPtr(iRow, iCol) == 1)) {
    
        // this implements a filter with potentially missing data, as specified by cMatPtr.  
        // disregard the entire row if either end pixel is missing
        // skip the center column because the coefficients are zero
        int rowMasks[3] = {
        cMatPtr(iRow-1, iCol-1)*cMatPtr(iRow-1, iCol+1), 
        cMatPtr(iRow,   iCol-1)*cMatPtr(iRow,   iCol+1), 
        cMatPtr(iRow+1, iCol-1)*cMatPtr(iRow+1, iCol+1)  };

        // this is the convolution of the gx kernel over the 3x3 pixel window
        // skip the center column because the coefficients are zero
        int sumgx = 
        gx[0]*rowMasks[0]*depthMat(iRow-1,iCol-1) + gx[2]*rowMasks[0]*depthMat(iRow-1,iCol+1) +
        gx[3]*rowMasks[1]*depthMat(iRow,  iCol-1) + gx[5]*rowMasks[1]*depthMat(iRow,  iCol+1) +
        gx[6]*rowMasks[2]*depthMat(iRow+1,iCol-1) + gx[8]*rowMasks[2]*depthMat(iRow+1,iCol+1);

        // disregard entire column if either end pixel is missing
        int colMasks[3] = {
            cMatPtr(iRow-1, iCol-1)*cMatPtr(iRow+1, iCol-1),  
            cMatPtr(iRow-1, iCol  )*cMatPtr(iRow+1, iCol),  
            cMatPtr(iRow-1, iCol+1)*cMatPtr(iRow+1, iCol+1)  };

        // this is the convolution of the gy kernel over the 3x3 pixel window
        // skip the center column because the coefficients are zero
        int sumgy = 
        gy[0]*colMasks[0]*depthMat(iRow-1,iCol-1) + gy[1]*colMasks[1]*depthMat(iRow-1,iCol) + gy[2]*colMasks[2]*depthMat(iRow-1,iCol+1) +
        gy[6]*colMasks[0]*depthMat(iRow+1,iCol-1) + gy[7]*colMasks[1]*depthMat(iRow+1,iCol) + gy[8]*colMasks[2]*depthMat(iRow+1,iCol+1);    

        float sumgx_f = __int2float_rn(sumgx);
        float sumgy_f = __int2float_rn(sumgy);
        float gradNorm = gradScale * sqrt(sumgx_f*sumgx_f + sumgy_f*sumgy_f);

        if (gradNorm > 0.5) {
           depthMatEdges(iRow, iCol) = 1; 

           // also set the converged flag for the pixel to 0
           cMatPtr(iRow, iCol) = 0;
        }
    }
}


//-----------------------------------------------------------------------------
__global__ void pointSmoothingKernel(cv::cuda::PtrStepSz<float3> posMatSmoothedPtr_m,  // outputs
                                     cv::cuda::PtrStepSz<float3> posMatPtr_m,           // inputs
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
__global__ void kalmanFilterKernel( cv::cuda::PtrStepSz<int> convergedMatPtr,  // outputs
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

    if (measurement_mm > 10.0) {
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
    if (depthMatVarPtr_mm2(iRow, iCol) < depthConvergedThresh_mm2){
        convergedMatPtr(iRow, iCol) = 1;
    }
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
// for realsense processing
__global__ void transformPositionToDepthKernel(cv::cuda::PtrStepSz<int> depthMatRotatedPtr_mm,   // outputs
                                            cv::cuda::PtrStepSz<int> depthShadedPtr_mm,
                                            cv::cuda::PtrStepSz<uchar> shadingMaskMatPtr,
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

    if (pointFrame1_m.z < 0.01f) {
        depthInRange = false;
    }

    pointFrame2_m = rotateTranslate(dcmDepthtoColor, posDepthToColor_m, pointFrame1_m);

    if (useRgbCoeff){
        pixelInRange = projectPointToPixel(pixelXY,         // outputs
                                            pointFrame2_m,  // inputs
                                            rgbPixelCoeffs,
                                            rgbLensCoeffs,
                                            width,
                                            height);
    } else {
        pixelInRange = projectPointToPixel(pixelXY,         // outputs
                                            pointFrame2_m,  // inputs
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
__global__ void generateShadedColorImageKernel(  cv::cuda::PtrStepSz<uchar3> shadedColorMatPtr,  // outputs
                                                 cv::cuda::PtrStepSz<uchar3> colorMatPtr,        // inputs
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

            // assume the transformation represents the depth to camera frames,
            // so look up the color in the newly computed color frame, and assign it to the
            // depth frame color
            color = colorMatPtr(mappedRow, mappedCol);

        }
        shadedColorMatPtr(iRow, iCol) = color;  
    }
}


//-----------------------------------------------------------------------------
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
__global__ void mapToOpenGLKernel(float3 *pointsOut,                        // outputs
                                  uchar3 *colorsOut,                              
                                  cv::cuda::PtrStepSz<int> depthMatPtr_mm,  // inputs
                                  cv::cuda::PtrStepSz<float> depthMatVarPtr_mm2,
                                  cv::cuda::PtrStepSz<float3> posMatPtr_m,
                                  cv::cuda::PtrStepSz<float3> normMatPtr,
                                  cv::cuda::PtrStepSz<uchar3> colorInDepthMatPtr,
                                  cv::cuda::PtrStepSz<float> pixelXPosMatPtr,
                                  cv::cuda::PtrStepSz<float> pixelYPosMatPtr,
                                  unsigned int width,
                                  unsigned int height)
{
    
    // inputs
    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    float3 point_m  = make_float3(0, 0, 0);
    uchar3 color = colorInDepthMatPtr(iRow, iCol);

    float3 normUvec = normMatPtr(iRow, iCol);
    float sum = normUvec.x*normUvec.x + normUvec.y*normUvec.y + normUvec.z*normUvec.z;

    // check to make sure variance of pixel is below a threshold
    if ( (depthMatVarPtr_mm2(iRow, iCol) <= depthConvergedThresh_mm2) &&
         (sum > 0) ) {

        //float z_m = __int2float_rn(depthMatPtr_mm(iRow, iCol)) * 0.001;
        //float x_m = pixelXPosMatPtr(iRow, iCol) * z_m;
        //float y_m = pixelYPosMatPtr(iRow, iCol) * z_m;
        //point_m = make_float3(x_m, -y_m, -z_m);  // for openGL, apply this transformation

        float3 tmp = posMatPtr_m(iRow, iCol);
        point_m = make_float3(tmp.x, -tmp.y, -tmp.z); 

    }

    // openGL and openCV have different RGB orders
    pointsOut[iRow * width + iCol] = point_m;
    colorsOut[iRow * width + iCol] = make_uchar3(color.z, color.y, color.x);
}


//-----------------------------------------------------------------------------
// this computes the matricies that need to have all their columns summed in order
// to solve the problem Ax = b
__global__ void computeLeastSquaresMatKernel( cv::cuda::PtrStepSz<float>  Bmat,            // outputs
                                              cv::cuda::PtrStepSz<float>  Dmat,
                                              cv::cuda::PtrStepSz<float>  residualMat,
                                              cv::cuda::PtrStepSz<uchar>  resMaskMat,
                                              cv::cuda::PtrStepSz<int>    depthMatEst_mm,  // inputs 
                                              cv::cuda::PtrStepSz<int>    depthMatMeas_mm,   
                                              cv::cuda::PtrStepSz<float3> normalMat,                            
                                              cv::cuda::PtrStepSz<float>  depthMatVar_mm2,     
                                              cv::cuda::PtrStepSz<float>  pixelXPosMat,
                                              cv::cuda::PtrStepSz<float>  pixelYPosMat,
                                              cv::cuda::PtrStepSz<int>    convergedMat,
                                              unsigned int width,
                                              unsigned int height){

    unsigned int iCol = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iRow = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int idx = iRow * width + iCol;

    float3 p = normalize(make_float3(pixelXPosMat(iRow,iCol), pixelYPosMat(iRow, iCol), 1.0f));
    float3 n = normalMat(iRow, iCol);

    float d1 = __int2float_rn(depthMatEst_mm(iRow, iCol)) * 0.001;
    float d2 = __int2float_rn(depthMatMeas_mm(iRow, iCol)) * 0.001;

    float npDot = dot(n, p);

    resMaskMat(iRow, iCol) = 0;

    if ( (d2 > 0.01) &&
         (npDot > angleThresh) && 
         (convergedMat(iRow, iCol) == 1) ) 
    {
        resMaskMat(iRow, iCol) = 1;

        float row[6] = { n.x, 
                         n.y, 
                         n.z,
                        -d2*(n.y*p.z - n.z*p.y),
                        -d2*(n.z*p.x - n.x*p.z),
                        -d2*(n.x*p.y - n.y*p.x)};

        float b = npDot * (d1 - d2);                        
                      
        Bmat(idx, 0)  = row[0]*row[0];
        Bmat(idx, 1)  = row[0]*row[1];
        Bmat(idx, 2)  = row[0]*row[2];
        Bmat(idx, 3)  = row[0]*row[3];
        Bmat(idx, 4)  = row[0]*row[4];
        Bmat(idx, 5)  = row[0]*row[5];
        Bmat(idx, 6)  = row[1]*row[1];
        Bmat(idx, 7)  = row[1]*row[2];
        Bmat(idx, 8)  = row[1]*row[3];
        Bmat(idx, 9)  = row[1]*row[4];
        Bmat(idx, 10) = row[1]*row[5];
        Bmat(idx, 11) = row[2]*row[2];
        Bmat(idx, 12) = row[2]*row[3];
        Bmat(idx, 13) = row[2]*row[4];
        Bmat(idx, 14) = row[2]*row[5];
        Bmat(idx, 15) = row[3]*row[3];
        Bmat(idx, 16) = row[3]*row[4];
        Bmat(idx, 17) = row[3]*row[5];
        Bmat(idx, 18) = row[4]*row[4];
        Bmat(idx, 19) = row[4]*row[5];
        Bmat(idx, 20) = row[5]*row[5];

        Dmat(idx, 0) = row[0] * b;
        Dmat(idx, 1) = row[1] * b;
        Dmat(idx, 2) = row[2] * b;
        Dmat(idx, 3) = row[3] * b;
        Dmat(idx, 4) = row[4] * b;
        Dmat(idx, 5) = row[5] * b;

        residualMat(iRow, iCol) = b;

    }
}  


//-----------------------------------------------------------------------------
__global__ void tallMatrixBlockReduction_8r(cv::cuda::PtrStepSz<float> reducedBmat,  // outputs
                                            cv::cuda::PtrStepSz<float> reducedDmat,
                                            cv::cuda::PtrStepSz<float> Bmat,         // inputs
                                            cv::cuda::PtrStepSz<float> Dmat,
                                            unsigned int matHeight){

      int iRow = blockIdx.x * blockDim.x + threadIdx.x;

      if (iRow < matHeight) {

        int idxB = iRow * 8;

        reducedBmat(iRow, 0) = Bmat(idxB,  0) + Bmat(idxB+1,  0) + Bmat(idxB+2,  0) + Bmat(idxB+3,  0) + Bmat(idxB+4,  0) + Bmat(idxB+5,  0) + Bmat(idxB+6,  0) + Bmat(idxB+7,  0);
        reducedBmat(iRow, 1) = Bmat(idxB,  1) + Bmat(idxB+1,  1) + Bmat(idxB+2,  1) + Bmat(idxB+3,  1) + Bmat(idxB+4,  1) + Bmat(idxB+5,  1) + Bmat(idxB+6,  1) + Bmat(idxB+7,  1);
        reducedBmat(iRow, 2) = Bmat(idxB,  2) + Bmat(idxB+1,  2) + Bmat(idxB+2,  2) + Bmat(idxB+3,  2) + Bmat(idxB+4,  2) + Bmat(idxB+5,  2) + Bmat(idxB+6,  2) + Bmat(idxB+7,  2);
        reducedBmat(iRow, 3) = Bmat(idxB,  3) + Bmat(idxB+1,  3) + Bmat(idxB+2,  3) + Bmat(idxB+3,  3) + Bmat(idxB+4,  3) + Bmat(idxB+5,  3) + Bmat(idxB+6,  3) + Bmat(idxB+7,  3);
        reducedBmat(iRow, 4) = Bmat(idxB,  4) + Bmat(idxB+1,  4) + Bmat(idxB+2,  4) + Bmat(idxB+3,  4) + Bmat(idxB+4,  4) + Bmat(idxB+5,  4) + Bmat(idxB+6,  4) + Bmat(idxB+7,  4);
        reducedBmat(iRow, 5) = Bmat(idxB,  5) + Bmat(idxB+1,  5) + Bmat(idxB+2,  5) + Bmat(idxB+3,  5) + Bmat(idxB+4,  5) + Bmat(idxB+5,  5) + Bmat(idxB+6,  5) + Bmat(idxB+7,  5);
        reducedBmat(iRow, 6) = Bmat(idxB,  6) + Bmat(idxB+1,  6) + Bmat(idxB+2,  6) + Bmat(idxB+3,  6) + Bmat(idxB+4,  6) + Bmat(idxB+5,  6) + Bmat(idxB+6,  6) + Bmat(idxB+7,  6);
        reducedBmat(iRow, 7) = Bmat(idxB,  7) + Bmat(idxB+1,  7) + Bmat(idxB+2,  7) + Bmat(idxB+3,  7) + Bmat(idxB+4,  7) + Bmat(idxB+5,  7) + Bmat(idxB+6,  7) + Bmat(idxB+7,  7);
        reducedBmat(iRow, 8) = Bmat(idxB,  8) + Bmat(idxB+1,  8) + Bmat(idxB+2,  8) + Bmat(idxB+3,  8) + Bmat(idxB+4,  8) + Bmat(idxB+5,  8) + Bmat(idxB+6,  8) + Bmat(idxB+7,  8);
        reducedBmat(iRow, 9) = Bmat(idxB,  9) + Bmat(idxB+1,  9) + Bmat(idxB+2,  9) + Bmat(idxB+3,  9) + Bmat(idxB+4,  9) + Bmat(idxB+5,  9) + Bmat(idxB+6,  9) + Bmat(idxB+7,  9);
        reducedBmat(iRow,10) = Bmat(idxB, 10) + Bmat(idxB+1, 10) + Bmat(idxB+2, 10) + Bmat(idxB+3, 10) + Bmat(idxB+4, 10) + Bmat(idxB+5, 10) + Bmat(idxB+6, 10) + Bmat(idxB+7, 10);
        reducedBmat(iRow,11) = Bmat(idxB, 11) + Bmat(idxB+1, 11) + Bmat(idxB+2, 11) + Bmat(idxB+3, 11) + Bmat(idxB+4, 11) + Bmat(idxB+5, 11) + Bmat(idxB+6, 11) + Bmat(idxB+7, 11);
        reducedBmat(iRow,12) = Bmat(idxB, 12) + Bmat(idxB+1, 12) + Bmat(idxB+2, 12) + Bmat(idxB+3, 12) + Bmat(idxB+4, 12) + Bmat(idxB+5, 12) + Bmat(idxB+6, 12) + Bmat(idxB+7, 12);
        reducedBmat(iRow,13) = Bmat(idxB, 13) + Bmat(idxB+1, 13) + Bmat(idxB+2, 13) + Bmat(idxB+3, 13) + Bmat(idxB+4, 13) + Bmat(idxB+5, 13) + Bmat(idxB+6, 13) + Bmat(idxB+7, 13);
        reducedBmat(iRow,14) = Bmat(idxB, 14) + Bmat(idxB+1, 14) + Bmat(idxB+2, 14) + Bmat(idxB+3, 14) + Bmat(idxB+4, 14) + Bmat(idxB+5, 14) + Bmat(idxB+6, 14) + Bmat(idxB+7, 14);
        reducedBmat(iRow,15) = Bmat(idxB, 15) + Bmat(idxB+1, 15) + Bmat(idxB+2, 15) + Bmat(idxB+3, 15) + Bmat(idxB+4, 15) + Bmat(idxB+5, 15) + Bmat(idxB+6, 15) + Bmat(idxB+7, 15);
        reducedBmat(iRow,16) = Bmat(idxB, 16) + Bmat(idxB+1, 16) + Bmat(idxB+2, 16) + Bmat(idxB+3, 16) + Bmat(idxB+4, 16) + Bmat(idxB+5, 16) + Bmat(idxB+6, 16) + Bmat(idxB+7, 16);
        reducedBmat(iRow,17) = Bmat(idxB, 17) + Bmat(idxB+1, 17) + Bmat(idxB+2, 17) + Bmat(idxB+3, 17) + Bmat(idxB+4, 17) + Bmat(idxB+5, 17) + Bmat(idxB+6, 17) + Bmat(idxB+7, 17);
        reducedBmat(iRow,18) = Bmat(idxB, 18) + Bmat(idxB+1, 18) + Bmat(idxB+2, 18) + Bmat(idxB+3, 18) + Bmat(idxB+4, 18) + Bmat(idxB+5, 18) + Bmat(idxB+6, 18) + Bmat(idxB+7, 18);
        reducedBmat(iRow,19) = Bmat(idxB, 19) + Bmat(idxB+1, 19) + Bmat(idxB+2, 19) + Bmat(idxB+3, 19) + Bmat(idxB+4, 19) + Bmat(idxB+5, 19) + Bmat(idxB+6, 19) + Bmat(idxB+7, 19);
        reducedBmat(iRow,20) = Bmat(idxB, 20) + Bmat(idxB+1, 20) + Bmat(idxB+2, 20) + Bmat(idxB+3, 20) + Bmat(idxB+4, 20) + Bmat(idxB+5, 20) + Bmat(idxB+6, 20) + Bmat(idxB+7, 20);

        reducedDmat(iRow, 0) = Dmat(idxB,  0) + Dmat(idxB+1,  0) + Dmat(idxB+2,  0) + Dmat(idxB+3,  0) + Dmat(idxB+4,  0) + Dmat(idxB+5,  0) + Dmat(idxB+6,  0) + Dmat(idxB+7,  0);
        reducedDmat(iRow, 1) = Dmat(idxB,  1) + Dmat(idxB+1,  1) + Dmat(idxB+2,  1) + Dmat(idxB+3,  1) + Dmat(idxB+4,  1) + Dmat(idxB+5,  1) + Dmat(idxB+6,  1) + Dmat(idxB+7,  1);
        reducedDmat(iRow, 2) = Dmat(idxB,  2) + Dmat(idxB+1,  2) + Dmat(idxB+2,  2) + Dmat(idxB+3,  2) + Dmat(idxB+4,  2) + Dmat(idxB+5,  2) + Dmat(idxB+6,  2) + Dmat(idxB+7,  2);
        reducedDmat(iRow, 3) = Dmat(idxB,  3) + Dmat(idxB+1,  3) + Dmat(idxB+2,  3) + Dmat(idxB+3,  3) + Dmat(idxB+4,  3) + Dmat(idxB+5,  3) + Dmat(idxB+6,  3) + Dmat(idxB+7,  3);
        reducedDmat(iRow, 4) = Dmat(idxB,  4) + Dmat(idxB+1,  4) + Dmat(idxB+2,  4) + Dmat(idxB+3,  4) + Dmat(idxB+4,  4) + Dmat(idxB+5,  4) + Dmat(idxB+6,  4) + Dmat(idxB+7,  4);
        reducedDmat(iRow, 5) = Dmat(idxB,  5) + Dmat(idxB+1,  5) + Dmat(idxB+2,  5) + Dmat(idxB+3,  5) + Dmat(idxB+4,  5) + Dmat(idxB+5,  5) + Dmat(idxB+6,  5) + Dmat(idxB+7,  5);
      }
}




///////////////////////////////////////////////////////////////////////////////
// kernel wrapper functions, called in .cpp files
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// the RGB camera is at a different location and orientation from the depth reciever
// compute an RGB image with pixels 1:1 with the depth image (as best as possible using given calibration coefficients)
extern "C"
void syncColorToDepthData( cv::cuda::GpuMat colorInDepthMat,       // outputs
                           cv::cuda::GpuMat depthMatRotated_mm,
                           cv::cuda::GpuMat depthMatShaded_mm,
                           cv::cuda::GpuMat shadingMaskMat,
                           cv::cuda::GpuMat xIdNewMat,
                           cv::cuda::GpuMat yIdNewMat,
                           cv::cuda::GpuMat posMat_m,
                           const cv::cuda::GpuMat depthMat_mm,    // inputs
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
                                                    shadingMaskMat,
                                                    xIdNewMat,
                                                    yIdNewMat,
                                                    posMat_m,           // inputs          
                                                    width,
                                                    height,
                                                    dcmDepthtoColor,
                                                    posDepthToColor_m,
                                                    useRgbCoeffs);

    // in this function, map the raw colorMat to the depth frame and store it in colorInDepthMat
    generateShadedColorImageKernel<<<grid, block>>>(  colorInDepthMat,     // outputs
                                                      colorMat,            // inputs
                                                      depthMatRotated_mm,   
                                                      depthMatShaded_mm,
                                                      shadingMaskMat,
                                                      xIdNewMat,
                                                      yIdNewMat,
                                                      width,
                                                      height);   
}


//-----------------------------------------------------------------------------
extern "C"
void transformData(cv::cuda::GpuMat colorInDepthMatTransformed,    // outputs
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
    depthToPositionKernel<<<grid, block>>>(posMat_m,   // ouputs           
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
extern "C"
void runDepthKalmanFilters(cv::cuda::GpuMat posMat_m,             // outputs
                           cv::cuda::GpuMat convergedMat,
                           cv::cuda::GpuMat depthMatBlurred,
                           cv::cuda::GpuMat depthMatEdges,
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

    kalmanFilterKernel<<<grid, block>>>(convergedMat,    // outputs
                                        depthMatEst_mm,  // states
                                        depthMatVar_mm2,
                                        depthMat_mm,     // inputs
                                        width,
                                        height); 

    GaussianBlur5x5Kernel<<<grid, block>>>(depthMatBlurred,  // outputs
                                           depthMatEst_mm,   // inputs
                                           convergedMat,
                                           width,
                                           height);

    edgeDetectorKernel<<<grid, block>>>(depthMatEdges,   // outputs
                                        convergedMat,
                                        depthMatEst_mm, // inputs
                                        width,
                                        height);

    depthMatBlurred.setTo(0);

    // blur again, this time ignoring the edge points when computing the convolution
    // GaussianBlur5x5Kernel<<<grid, block>>>(depthMatBlurred,  // outputs
    //                                        depthMatEst_mm,   // inputs
    //                                        convergedMat,
    //                                        width,
    //                                        height);

    // depthMatEst_mm = depthMatBlurred.clone();

    depthToPositionKernel<<<grid, block>>>(posMat_m,       // ouputs           
                                           depthMatEst_mm,  // inputs
                                           pixelXPosMat,  
                                           pixelYPosMat,
                                           width,
                                           height); 

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

    computeNormalKernel<<<grid, block>>>(normMat,          // outputs
                                         posMat_m,         // inputs
                                         convergedMat,
                                         pixelXPosMat,
                                         pixelYPosMat,
                                         width,
                                         height);                                                                                 
}


//-----------------------------------------------------------------------------
// the RGB camera is at a different location and orientation from the depth reciever
// compute an RGB image with pixels 1:1 with the depth image (as best as possible using given calibration coefficients)
extern "C"
void mapColorDepthToOpenGL(float3 *pointsOut,                     // outputs
                           uchar3 *colorsOut,
                           const cv::cuda::GpuMat depthMat_mm,    // inputs
                           const cv::cuda::GpuMat depthMatVar_mm2,
                           const cv::cuda::GpuMat posMat_m,
                           const cv::cuda::GpuMat normMat,
                           const cv::cuda::GpuMat colorInDepthMat,
                           const cv::cuda::GpuMat pixelXPosMat,
                           const cv::cuda::GpuMat pixelYPosMat,
                           const unsigned int width,
                           const unsigned int height)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // map the vertex 3D points and color mat to openGL VBOs in the main.cpp
    mapToOpenGLKernel<<<grid, block>>>(pointsOut,       // outputs
                                       colorsOut,                              
                                       depthMat_mm,     // inputs
                                       depthMatVar_mm2,
                                       posMat_m,
                                       normMat,
                                       colorInDepthMat,
                                       pixelXPosMat,
                                       pixelYPosMat,
                                       width,
                                       height);   
}


//-----------------------------------------------------------------------------
extern "C"
void computeLeastSqauresMat(cv::cuda::GpuMat Bmat,             // outputs
                            cv::cuda::GpuMat reducedBmat1,
                            cv::cuda::GpuMat reducedBmat2,
                            cv::cuda::GpuMat reducedBmat3,
                            cv::cuda::GpuMat Dmat,
                            cv::cuda::GpuMat reducedDmat1,
                            cv::cuda::GpuMat reducedDmat2,
                            cv::cuda::GpuMat reducedDmat3, 
                            cv::cuda::GpuMat residualMat,   
                            cv::cuda::GpuMat resMaskMat,
                            cv::cuda::GpuMat depthMatEst_mm,    // inputs
                            cv::cuda::GpuMat depthMat_mm,      
                            cv::cuda::GpuMat normalMat,        
                            cv::cuda::GpuMat depthMatVar_mm2, 
                            cv::cuda::GpuMat pixelXPosMat,    
                            cv::cuda::GpuMat pixelYPosMat,
                            cv::cuda::GpuMat convergedMat,
                            const unsigned int width,
                            const unsigned int height)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // compute the large least squares matricies
    computeLeastSquaresMatKernel<<<grid, block>>>( Bmat,            // outputs
                                                   Dmat,
                                                   residualMat,
                                                   resMaskMat,
                                                   depthMatEst_mm,  // inputs 
                                                   depthMat_mm,
                                                   normalMat,                             
                                                   depthMatVar_mm2,     
                                                   pixelXPosMat,
                                                   pixelYPosMat,
                                                   convergedMat,
                                                   width,
                                                   height);   

    // do sum block reduction to cut them down to size by summing the columns
    // the final sumation will be done on the cpu, but the matricies will be much smaller
    int n = width * height;

    int blockSize = 64;
    // 50880
    int matHeight = (n / 8);
    // 795
    int numBlocks = matHeight / blockSize;

    tallMatrixBlockReduction_8r<<<numBlocks, blockSize>>>(reducedBmat1,  // outputs
                                                          reducedDmat1,    
                                                          Bmat,          // inputs 
                                                          Dmat,          
                                                          matHeight);

    blockSize = 32;
    // 6360
    matHeight = (n / 64);
    numBlocks = 200;

    tallMatrixBlockReduction_8r<<<numBlocks, blockSize>>>(reducedBmat2,  // outputs
                                                          reducedDmat2,    
                                                          reducedBmat1,  // inputs 
                                                          reducedDmat1,          
                                                          matHeight);

    blockSize = 32;
    // 795
    matHeight = (n / 512);
    numBlocks = 25;

    tallMatrixBlockReduction_8r<<<numBlocks, blockSize>>>(reducedBmat3,  // outputs
                                                          reducedDmat3,    
                                                          reducedBmat2,  // inputs 
                                                          reducedDmat2,          
                                                          matHeight);

}

