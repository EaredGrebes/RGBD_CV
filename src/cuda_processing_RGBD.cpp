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
void syncColorToDepthData( cv::cuda::GpuMat colorInDepthMat,     // outputs
                           cv::cuda::GpuMat depthMatRotated_mm,
                           cv::cuda::GpuMat depthMatShaded_mm,
                           cv::cuda::GpuMat MaskMat,
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
void transformData(cv::cuda::GpuMat colorInDepthMatTransformed,    // outputs
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
                           const unsigned int height);

extern "C"
void mapColorDepthToOpenGL(float3 *pointsOut,                     // outputs
                           uchar3 *colorsOut,
                           const cv::cuda::GpuMat depthMat_mm,    // inputs
                           const cv::cuda::GpuMat depthMatVar_mm2,
                           const cv::cuda::GpuMat posMat_mm,
                           const cv::cuda::GpuMat normMat,
                           const cv::cuda::GpuMat colorInDepthMat,
                           const cv::cuda::GpuMat pixelXPosMat,
                           const cv::cuda::GpuMat pixelYPosMat,
                           const unsigned int width,
                           const unsigned int height);

extern "C"
void computeLeastSqauresMat(cv::cuda::GpuMat Bmat,            // outputs
                            cv::cuda::GpuMat reducedBmat1,
                            cv::cuda::GpuMat reducedBmat2,
                            cv::cuda::GpuMat reducedBmat3,
                            cv::cuda::GpuMat Dmat,
                            cv::cuda::GpuMat reducedDmat1,
                            cv::cuda::GpuMat reducedDmat2,
                            cv::cuda::GpuMat reducedDmat3,
                            cv::cuda::GpuMat residualMat,
                            cv::cuda::GpuMat resMaskMat,
                            cv::cuda::GpuMat depthMatEst_mm, // inputs
                            cv::cuda::GpuMat depthMat_mm,
                            cv::cuda::GpuMat normalMat,      
                            cv::cuda::GpuMat depthMatVar_mm2, 
                            cv::cuda::GpuMat pixelXPosMat,    
                            cv::cuda::GpuMat pixelYPosMat,
                            cv::cuda::GpuMat convergedMat,
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

    xOpt <<  0, 0, 0, 0.0, -0.0,  0.0;
    dxLim << 2, 2, 2, 0.2, 0.2, 0.2;
    dx << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01;
    xScale << 0.007, 0.007, 0.007, 0.0001, 0.0001, 0.0001;


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
    shadingMaskMat.create(cv::Size(meshWidth, meshHeight), CV_8U);
    
    colorInDepthMat.create(cv::Size(meshWidth, meshHeight), CV_8UC3);

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


    // ~~ for solving transform problem ~~ //
    residualMat.create(cv::Size(meshWidth, meshHeight), CV_32F);
    resMaskMat.create(cv::Size(meshWidth, meshHeight), CV_8U);

    // width = 21 is the number of independent parameters in the 6x6 symetrix matrix A' * A
    // height = n = width * height because we have at maximum 1 measurement point for each pixel
    int n = meshWidth*meshHeight;
    Bmat.create(cv::Size(21, n), CV_32F);
    Dmat.create(cv::Size(6,  n), CV_32F);

    // use 8 row at-a-time, in 3 stages to sum each column of the Bmat and Dmat
    reducedBmat1.create(cv::Size(21, n/8),   CV_32F);
    reducedBmat2.create(cv::Size(21, n/64),  CV_32F);
    reducedBmat3.create(cv::Size(21, n/512), CV_32F);

    reducedDmat1.create(cv::Size(6, n/8),   CV_32F);
    reducedDmat2.create(cv::Size(6, n/64),  CV_32F);
    reducedDmat3.create(cv::Size(6, n/512), CV_32F);

    reducedBmat3Cpu.create(cv::Size(21, n/512), CV_32F);
    reducedDmat3Cpu.create(cv::Size(6, n/512), CV_32F);

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
    // But having a variance estimate for each pixel is important later on in the
    // SLAM algorithm.  
    depthMatEst_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);
    depthMatVar_mm2.create(cv::Size(meshWidth, meshHeight), CV_32F);

    depthMatEst_mm.setTo(initDepthEstimate_mm);
    depthMatVar_mm2.setTo(initDepthVar_mm2);
}

//------------------------------------------------------------------------------
// for visualizing the output of the RGBD data, this puts a 
void cuda_processing_RGBD::computeVertexAndColorPoints(float3 *pointsOut,    // outputs
                                                       uchar3 *colorsOut) {

  // // solve linear system for translation and rotation 
  residualMat.setTo(0.0f);

  computeLeastSqauresMat(Bmat,            // outputs
                        reducedBmat1,
                        reducedBmat2,
                        reducedBmat3,
                        Dmat,
                        reducedDmat1,
                        reducedDmat2,
                        reducedDmat3,
                        residualMat,
                        resMaskMat,
                        depthMatEst_mm,  // inputs
                        depthMat_mm,
                        normalMat,       
                        depthMatVar_mm2, 
                        pixelXPosMat,    
                        pixelYPosMat,
                        convergedMat,
                        meshWidth,
                        meshHeight);

  reducedBmat3.download(reducedBmat3Cpu);
  reducedDmat3.download(reducedDmat3Cpu);

  cv::Mat BcolSum, DcolSum;
  cv::reduce(reducedBmat3Cpu, BcolSum, 0, cv::REDUCE_SUM, CV_32F);
  cv::reduce(reducedDmat3Cpu, DcolSum, 0, cv::REDUCE_SUM, CV_32F);

  Eigen::Matrix<float,6,1> x;

  solveAxb(x,        // outputs
           BcolSum,  // inputs
           DcolSum);

  colorInDepthMatTransformed.setTo(cv::Scalar(200, 0, 200));
  depthMatTransformed_mm.setTo(depthMax_mm);
  shadingMaskMat.setTo(0);
  depthMatVarTransformed_mm2.setTo(initDepthVar_mm2);
  posMat_m.setTo(0);

  transformData(colorInDepthMatTransformed, // outputs
               depthMatVarTransformed_mm2,
               depthMatTransformed_mm,
               depthMatRotated_mm,
               shadingMaskMat,
               xIdNewMat,
               yIdNewMat,
               posMat_m,   
               depthMatEst_mm,               // inputs
               depthMatVar_mm2,            
               colorInDepthMat,
               depthMat_mm,
               pixelXPosMat,
               pixelYPosMat,
               meshWidth,
               meshHeight,
               dcmRotation,
               posTranslation_m);


  depthMatEst_mm = depthMatTransformed_mm.clone();                  
  depthMatVar_mm2 = depthMatVarTransformed_mm2.clone();      

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
    colorInDepthMat.setTo(cv::Scalar(200, 0, 200));

    // // for the depth shading to work, this matrix must have initial depth values far away
    depthMatShaded_mm.setTo(depthMax_mm);

    // // set the masking vector for the shaded pixels to all 0
    shadingMaskMat.setTo(0);

    // sync the most recent color image to the updated
    // depth estimates
    syncColorToDepthData(colorInDepthMat,     // outputs
                        depthMatRotated_mm,
                        depthMatShaded_mm,
                        shadingMaskMat, 
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

    mapColorDepthToOpenGL( pointsOut,      // outputs
                           colorsOut,
                           depthMatEst_mm,    // inputs
                           depthMatVar_mm2,
                           posMat_m,
                           normalMat,
                           colorInDepthMat,
                           pixelXPosMat,
                           pixelYPosMat,
                           meshWidth,
                           meshHeight);
}

//--------------------------------------------------------------
void cuda_processing_RGBD::solveAxb(Eigen::Matrix<float,6,1> &x,  // outputs
                                    cv::Mat &smB,              // inputs
                                    cv::Mat &smD)
{

  // compute the rotation and translation from the estimated depth output
  // from the last step, and the raw sensor depth image of the current step

  float *p = (float *)smB.data;
  float *p2 = (float *)smD.data;

  Eigen::Matrix<float,6,6> A {{p[0], p[1],  p[2],   p[3],   p[4],   p[5]},
                              {p[1], p[6],  p[7],   p[8],   p[9],   p[10]},
                              {p[2], p[7],  p[11],  p[12],  p[13],  p[14]},
                              {p[3], p[8],  p[12],  p[15],  p[16],  p[17]},
                              {p[4], p[9],  p[13],  p[16],  p[18],  p[19]},
                              {p[5], p[10], p[14],  p[17],  p[19],  p[20]} };
                            
  
  Eigen::Vector<float, 6> b( p2[0], p2[1], p2[2], p2[3], p2[4], p2[5]);

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

  x = svd.solve(b);
  std::cout << "x: " << x << std::endl;

  float scale = -0.3;
  x = x * scale;

  Eigen::Vector<float, 3> r1( x(0), x(1), x(2) );

  Eigen::Matrix<float,3,3> dcm{{ 1,    -x(5),  x(4)},
                               { x(5),  1,    -x(3)},
                               {-x(4),  x(3),  1}};                             

  Eigen::JacobiSVD<Eigen::MatrixXf> svd2(dcm, Eigen::ComputeFullU | Eigen::ComputeFullV);    
  
  dcm = svd2.matrixU() * svd2.matrixV().transpose();



  Eigen::Vector<float, 3> r2 = dcm * r1;

  // use the computed transformation to transform the estimated states
  float rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float trans[3] = {r2(0), r2(1), r2(2)};

  for (unsigned int ii = 0; ii < 3; ii ++){
      for (unsigned int jj = 0; jj < 3; jj ++){
          rot[ii + jj*3] = dcm(ii,jj);
      }
  }

  cudaMemcpy(dcmRotation,      rot,   9*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(posTranslation_m, trans, 3*sizeof(float), cudaMemcpyHostToDevice);                
}


//--------------------------------------------------------------
// given a state vector x, compute the apriori distribution of 
// the obersvation vector y (which is the depth of all the pixels)
// then compute the likelihood of the measurement y
void cuda_processing_RGBD::computeLikelihoodMat(const Eigen::Matrix<float,6,1> &xIn) 
{
  colorInDepthMatTransformed.setTo(cv::Scalar(200, 0, 200));
  depthMatTransformed_mm.setTo(depthMax_mm);
  shadingMaskMat.setTo(0);
  depthMatVarTransformed_mm2.setTo(initDepthVar_mm2);
  posMat_m.setTo(0);

  stateVecToGpu(xIn);

  transformData(colorInDepthMatTransformed, // outputs
               depthMatVarTransformed_mm2,
               depthMatTransformed_mm,
               depthMatRotated_mm,
               shadingMaskMat,
               xIdNewMat,
               yIdNewMat,
               posMat_m,   
               depthMatEst_mm,               // inputs
               depthMatVar_mm2,            
               colorInDepthMat,
               depthMat_mm,
               pixelXPosMat,
               pixelYPosMat,
               meshWidth,
               meshHeight,
               dcmRotation,
               posTranslation_m);
}


//--------------------------------------------------------------
void cuda_processing_RGBD::stateVecToGpu(const Eigen::Matrix<float,Nopt,1> &xIn)           // inputs
{

  Eigen::Matrix3f dcm;
  dcm = Eigen::AngleAxisf(xIn(2,0)*M_PI/180, Eigen::Vector3f::UnitZ())
      * Eigen::AngleAxisf(xIn(1,0)*M_PI/180, Eigen::Vector3f::UnitY())
      * Eigen::AngleAxisf(xIn(0,0)*M_PI/180, Eigen::Vector3f::UnitX());


  float rot[9];
  float trans[3];

  for (unsigned int ii = 0; ii < 3; ii ++){
      for (unsigned int jj = 0; jj < 3; jj ++){

          rot[ii + jj*3] = dcm(ii,jj);
      }
      trans[ii] = xIn(ii+3,0);
  }

  cudaMemcpy(dcmRotation,      rot,   9*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(posTranslation_m, trans, 3*sizeof(float), cudaMemcpyHostToDevice);
}


//--------------------------------------------------------------
cuda_processing_RGBD::~cuda_processing_RGBD() {

}

