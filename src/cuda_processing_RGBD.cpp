// C++ stuff
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// opencv
#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudaarithm.hpp>

// CUDA
#include <cuda_runtime.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "cuda_processing_RGBD.h"


////////////////////////////////////////////////////////////////////////////////
// cuda kernel wrapper function calls
////////////////////////////////////////////////////////////////////////////////
// defined in <cuda_kernel.cu>

extern "C"
void syncColorToDepthData(cv::cuda::GpuMat colorInDepthMat_r,   // outputs
                          cv::cuda::GpuMat colorInDepthMat_g,
                          cv::cuda::GpuMat colorInDepthMat_b,
                          cv::cuda::GpuMat clrInDepthMatBlurred_r,   
                          cv::cuda::GpuMat clrInDepthMatBlurred_g,
                          cv::cuda::GpuMat clrInDepthMatBlurred_b,
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
                          const unsigned int mesh_width,
                          const unsigned int mesh_height,
                          float *dcmDepthtoColor,
                          float *posDepthToColor_m);   

extern "C"
void transformData(cv::cuda::GpuMat colorInDepthMatTransformed,  // outputs
                   cv::cuda::GpuMat depthMatVarTransformed_mm2,
                   cv::cuda::GpuMat depthMatTransformed_mm,
                   cv::cuda::GpuMat depthMatRotated_mm,
                   cv::cuda::GpuMat MaskMat,
                   cv::cuda::GpuMat xIdNewMat,
                   cv::cuda::GpuMat yIdNewMat,
                   const cv::cuda::GpuMat posMat_m,                
                   const cv::cuda::GpuMat depthMat_mm,           // inputs
                   const cv::cuda::GpuMat depthMatVar_mm2,
                   const cv::cuda::GpuMat colorMatInput,  
                   const cv::cuda::GpuMat depthMatMeas_mm,                     
                   const cv::cuda::GpuMat pixelXPosMat,
                   const cv::cuda::GpuMat pixelYPosMat,
                   const unsigned int width,
                   const unsigned int height,
                   float *dcmRotation,
                   float *posTrans_m);

extern "C"
void runDepthKalmanFilters(cv::cuda::GpuMat posMat_m,            // outputs
                           cv::cuda::GpuMat convergedMat, 
                           cv::cuda::GpuMat depthMatBlurred,
                           cv::cuda::GpuMat depthMatEdges,
                           cv::cuda::GpuMat posMatSmoothed_m,
                           cv::cuda::GpuMat normMat,
                           cv::cuda::GpuMat depthMatEst_mm,      // states
                           cv::cuda::GpuMat depthMatVar_mm2,
                           const cv::cuda::GpuMat depthMat_mm,   // inputs
                           const cv::cuda::GpuMat pixelXPosMat,
                           const cv::cuda::GpuMat pixelYPosMat,
                           const unsigned int width,
                           const unsigned int height);

extern "C"
void mapColorDepthToOpenGL(float3 *pointsOut,                     // outputs
                           uchar3 *colorsOut,
                           const cv::cuda::GpuMat depthMat_mm,    // inputs
                           const cv::cuda::GpuMat depthMatVar_mm2,
                           const cv::cuda::GpuMat posMat_mm,
                           const cv::cuda::GpuMat normMat,
                           const cv::cuda::GpuMat colorInDepthMat_r,
                           const cv::cuda::GpuMat colorInDepthMat_g,
                           const cv::cuda::GpuMat colorInDepthMat_b,
                           const cv::cuda::GpuMat pixelXPosMat,
                           const cv::cuda::GpuMat pixelYPosMat,
                           const unsigned int width,
                           const unsigned int height);
                       


////////////////////////////////////////////////////////////////////////////////
// class methods
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
cuda_processing_RGBD::cuda_processing_RGBD(int meshWidth_in, 
                                           int meshHeight_in,
                                           float *dcmDepthtoColorIn,
                                           float *posDepthToColorIn_m) {

    // ~~ parameters ~~ //
    meshWidth = meshWidth_in;
    meshHeight = meshHeight_in;

    depthMax_mm = 200 * 1000;  // 200 meters

    initDepthEstimate_mm = 2000;
    initDepthVar_mm2 = 5000 * 5000;

    // for converting the depth image to 3D points
    //
    // x = pixelXLocMat(i,j) * depthMat(i,j)
    // y = pixelYLocMat(i,j) * depthMat(i,j)
    // z = depthMat(i,j)
    //
    // remember cv::Size is flipped, so these matricies are really (height, width)
    pixelXPosMat.create(cv::Size(meshWidth, meshHeight), CV_32F);
    pixelYPosMat.create(cv::Size(meshWidth, meshHeight), CV_32F);

    // realsense calibration parameters
    cudaMalloc(&dcmDepthtoColor,   9*sizeof(float)); 
    cudaMalloc(&posDepthToColor_m, 3*sizeof(float));

    cudaMemcpy(dcmDepthtoColor,   dcmDepthtoColorIn,   9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(posDepthToColor_m, posDepthToColorIn_m, 3*sizeof(float), cudaMemcpyHostToDevice);

    // ~~ inputs ~~ //
    depthMat_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);
    colorMat.create(cv::Size(meshWidth, meshHeight), CV_8UC3);

    // ~~ RGBD image processing ~~ //
    depthMatRotated_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);
    depthMatShaded_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);
    depthShadedMaskMat.create(cv::Size(meshWidth, meshHeight), CV_32S);
    clrShadedMaskMat.create(cv::Size(meshWidth, meshHeight), CV_32S);

    colorInDepthMat_r.create(cv::Size(meshWidth, meshHeight), CV_32S);
    colorInDepthMat_g.create(cv::Size(meshWidth, meshHeight), CV_32S);
    colorInDepthMat_b.create(cv::Size(meshWidth, meshHeight), CV_32S);

    clrInDepthMatBlurred_r.create(cv::Size(meshWidth, meshHeight), CV_32S);
    clrInDepthMatBlurred_g.create(cv::Size(meshWidth, meshHeight), CV_32S);
    clrInDepthMatBlurred_b.create(cv::Size(meshWidth, meshHeight), CV_32S);    

    xIdNewMat.create(cv::Size(meshWidth, meshHeight), CV_32S);
    yIdNewMat.create(cv::Size(meshWidth, meshHeight), CV_32S);

    posMat_m.create(cv::Size(meshWidth, meshHeight), CV_32FC3);
    posMatSmoothed_m.create(cv::Size(meshWidth, meshHeight), CV_32FC3);

    depthMatTransformed_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);
    colorInDepthMatTransformed.create(cv::Size(meshWidth, meshHeight), CV_8UC3);
    depthMatVarTransformed_mm2.create(cv::Size(meshWidth, meshHeight), CV_32F);

    convergedMat.create(cv::Size(meshWidth, meshHeight), CV_32S);
    depthMatBlurred.create(cv::Size(meshWidth, meshHeight), CV_32S);  // for edge detection
    depthMatEdges.create(cv::Size(meshWidth, meshHeight), CV_32F);
    normalMat.create(cv::Size(meshWidth, meshHeight), CV_32FC3);

    // transform: rotation and translation
    float rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float trans[3] = {0.0, 0.0, 0.0};

    cudaMalloc(&dcmRotation,   9*sizeof(float)); 
    cudaMalloc(&posTranslation_m, 3*sizeof(float));

    cudaMemcpy(dcmRotation,   rot,   9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(posTranslation_m, trans, 3*sizeof(float), cudaMemcpyHostToDevice);

    // kalman filter for each depth pixel.  This is boring, as it assumes no structure
    // in the image (no correlation with other pixels), which isn't true, but it keeps the
    // dimensionality to 2*(width*height) rather than (width*height) + ((width*height)^2)
    // for a full covariance matrix for the whole image.  
    //
    // But having a variance estimate for each pixel is important later on 
    depthMatEst_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);
    depthMatVar_mm2.create(cv::Size(meshWidth, meshHeight), CV_32F);

    depthMatEst_mm.setTo(initDepthEstimate_mm);
    depthMatVar_mm2.setTo(initDepthVar_mm2);
}

//------------------------------------------------------------------------------
// for visualizing the output of the RGBD data, this puts a 
void cuda_processing_RGBD::runCudaProcessing(float3 *pointsOut,    // outputs
                                             uchar3 *colorsOut) {     

    // update the transformed kalman filter states with the most
    // recent depth image
    convergedMat.setTo(0);
    depthMatBlurred.setTo(0);
    depthMatEdges.setTo(0);
    normalMat.setTo(0);
    posMat_m.setTo(0);

    runDepthKalmanFilters(posMat_m,        // outputs
                          convergedMat,
                          depthMatBlurred,
                          depthMatEdges,
                          posMatSmoothed_m,
                          normalMat,
                          depthMatEst_mm,  // states
                          depthMatVar_mm2,
                          depthMat_mm,     // inputs
                          pixelXPosMat,
                          pixelYPosMat,
                          meshWidth,
                          meshHeight);

    // initialize all points to magenta, that way 3-d points that are shaded 
    // and don't have a color will appear visible, figure out how to handle (or fill in) missing color info later.
    colorInDepthMat_r.setTo(200);
    colorInDepthMat_g.setTo(0);
    colorInDepthMat_b.setTo(0);

    // // for the depth shading to work, this matrix must have initial depth values far away
    depthMatShaded_mm.setTo(depthMax_mm);

    // // set the masking vector for the shaded pixels to all 0
    depthShadedMaskMat.setTo(0);
    clrShadedMaskMat.setTo(0);

    // sync the most recent color image to the updated
    // depth estimates
    syncColorToDepthData(colorInDepthMat_r,  // outputs
                         colorInDepthMat_g,
                         colorInDepthMat_b,
                         clrInDepthMatBlurred_r,
                         clrInDepthMatBlurred_g,
                         clrInDepthMatBlurred_b,
                         depthMatRotated_mm,
                         depthMatShaded_mm,
                         depthShadedMaskMat, 
                         clrShadedMaskMat,
                         xIdNewMat,
                         yIdNewMat,
                         posMat_m,
                         depthMatEst_mm,      // input
                         colorMat,
                         pixelXPosMat,
                         pixelYPosMat,
                         meshWidth,
                         meshHeight,
                         dcmDepthtoColor,
                         posDepthToColor_m);

    mapColorDepthToOpenGL(pointsOut,      // outputs
                          colorsOut,
                          depthMatEst_mm, // inputs
                          depthMatVar_mm2,
                          posMat_m,
                          normalMat,
                          clrInDepthMatBlurred_r,
                          clrInDepthMatBlurred_g,
                          clrInDepthMatBlurred_b,
                          pixelXPosMat,
                          pixelYPosMat,
                          meshWidth,
                          meshHeight);
}


//--------------------------------------------------------------
cuda_processing_RGBD::~cuda_processing_RGBD() {

}

