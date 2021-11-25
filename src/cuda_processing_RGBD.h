#ifndef SRC_CUDAPROCESSINGRGBD_H_
#define SRC_CUDAPROCESSINGRGBD_H_

// C++ stuff
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// opencv
#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"

// CUDA
#include <cuda_runtime.h>

// Eigen
#include <Eigen/Dense>

const int Nopt = 6;


class cuda_processing_RGBD {
public:
	cuda_processing_RGBD(int meshWidth_in, 
		                 int meshHeight_in,
		                 float *dcmDepthtoColorIn,
                         float *posDepthToColorIn_m);

	virtual ~cuda_processing_RGBD();

	// ~~ parameters ~~ //
	cv::cuda::GpuMat pixelXPosMat;
	cv::cuda::GpuMat pixelYPosMat;

	int meshWidth;
	int meshHeight;

	float *dcmDepthtoColor;
	float *posDepthToColor_m;

	int initDepthEstimate_mm;
	float initDepthVar_mm2;
	int depthMax_mm;

	Eigen::Matrix<float,Nopt,1> dxLim;    // for optimization problem
	Eigen::Matrix<float,Nopt,1> dx;       // for optimization problem
	Eigen::Matrix<float,Nopt,1> xScale;   // for optimization problem

	// ~~ inputs ~~ //
	cv::cuda::GpuMat colorMat;
	cv::cuda::GpuMat depthMat_mm;
	cv::cuda::GpuMat depthMatPrev_mm;

	// ~~ states ~~ //

	// for transforming shading RGBD images
	cv::cuda::GpuMat depthMatRotated_mm;
	cv::cuda::GpuMat depthMatShaded_mm;
	cv::cuda::GpuMat shadingMaskMat;
	cv::cuda::GpuMat colorInDepthMat;

	cv::cuda::GpuMat xIdNewMat;
	cv::cuda::GpuMat yIdNewMat;

	cv::cuda::GpuMat depthMatTransformed_mm;
	cv::cuda::GpuMat colorInDepthMatTransformed;
	cv::cuda::GpuMat depthMatVarTransformed_mm2;

	cv::cuda::GpuMat posMat_m;
	cv::cuda::GpuMat posMatSmoothed_m;

	Eigen::Matrix<float,Nopt,1> xOpt;    // for optimization problem

	// kalman filters for each depth pixel
	cv::cuda::GpuMat depthMatEst_mm;
	cv::cuda::GpuMat depthMatVar_mm2;

	// ~~ outputs ~~ //
	cv::cuda::GpuMat convergedMat;
	cv::cuda::GpuMat normalMat;
	cv::cuda::GpuMat depthMatBlurred;
	cv::cuda::GpuMat depthMatEdges;

	// for solving the transformation optimization problem
	cv::cuda::GpuMat Bmat;
	cv::cuda::GpuMat Dmat;
	cv::cuda::GpuMat residualMat;
	cv::cuda::GpuMat resMaskMat;

	cv::cuda::GpuMat reducedBmat1;
	cv::cuda::GpuMat reducedBmat2;
	cv::cuda::GpuMat reducedBmat3;

	cv::cuda::GpuMat reducedDmat1;
	cv::cuda::GpuMat reducedDmat2;
	cv::cuda::GpuMat reducedDmat3;

	cv::Mat reducedBmat3Cpu;
	cv::Mat reducedDmat3Cpu;

	float *dcmRotation;
	float *posTranslation_m;

	Eigen::Matrix<float,Nopt,1> gradient;    // for optimization problem


	void computeVertexAndColorPoints(float3 *pointsOut,  // outputs
		                             uchar3 *colorsOut); 

	void solveAxb(Eigen::Matrix<float,6,1> &x,  // outputs
		          cv::Mat &sumBmat,              // inputs
		          cv::Mat &sumDmat);

	void computeLikelihoodMat(const Eigen::Matrix<float,6,1> &xIn);

	void stateVecToGpu(const Eigen::Matrix<float,Nopt,1> &xIn);  // inputs


private:


};

#endif /* SRC_CUDAPROCESSINGRGBD_H_ */
