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

	// ~~ inputs ~~ //
	cv::cuda::GpuMat colorMat;
	cv::cuda::GpuMat depthMat_mm;
	cv::cuda::GpuMat depthMatPrev_mm;

	// ~~ states ~~ //
    // kalman filters for each depth pixel
	cv::cuda::GpuMat depthMatEst_mm;
	cv::cuda::GpuMat depthMatVar_mm2;

	// for transforming shading RGBD images
	cv::cuda::GpuMat depthMatRotated_mm;
	cv::cuda::GpuMat depthMatShaded_mm;
	cv::cuda::GpuMat depthShadedMaskMat;
	cv::cuda::GpuMat clrShadedMaskMat;
	cv::cuda::GpuMat clrShadedMaskBlurMat;
	cv::cuda::GpuMat colorInDepthMat;

	cv::cuda::GpuMat colorInDepthMat_r;  // why split these into rgb? because I can't figure out how to loop through vector components in cuda (.x, .y, .z)
	cv::cuda::GpuMat colorInDepthMat_g;
	cv::cuda::GpuMat colorInDepthMat_b;

	cv::cuda::GpuMat clrInDepthMatBlurred_r;  
	cv::cuda::GpuMat clrInDepthMatBlurred_g;
	cv::cuda::GpuMat clrInDepthMatBlurred_b;

	cv::cuda::GpuMat xIdNewMat;
	cv::cuda::GpuMat yIdNewMat;

	cv::cuda::GpuMat depthMatTransformed_mm;
	cv::cuda::GpuMat colorInDepthMatTransformed;
	cv::cuda::GpuMat depthMatVarTransformed_mm2;

	cv::cuda::GpuMat posMat_m;
	cv::cuda::GpuMat posMatSmoothed_m;

	// ~~ outputs ~~ //
	cv::cuda::GpuMat convergedMat;
	cv::cuda::GpuMat normalMat;
	cv::cuda::GpuMat depthMatBlurred;
	cv::cuda::GpuMat depthMatEdges;

	float *dcmRotation;
	float *posTranslation_m;

	void runCudaProcessing( float3 *pointsOut,  // outputs
                            uchar3 *colorsOut); 									                   

private:


};

#endif /* SRC_CUDAPROCESSINGRGBD_H_ */
