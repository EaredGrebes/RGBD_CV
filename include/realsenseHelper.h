#ifndef SRC_REALSENSEHELPER_H_
#define SRC_REALSENSEHELPER_H_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>
#include <opencv2/core.hpp>


class realsenseHelper {
public:
	realsenseHelper(int meshWidth, int meshHeight);
	virtual ~realsenseHelper();

	std::string get_device_name(const rs2::device& dev);
	void print_device_information(const rs2::device& dev);
	rs2::sensor get_a_sensor_from_a_device(const rs2::device& dev, int selected_sensor_index);
	std::string get_sensor_name(const rs2::sensor& sensor);
	rs2::pipeline get_pipe();
	bool checkForNewFrame();


	unsigned int widthPixels;
	unsigned int heightPixels;

	cv::Mat colorMat;
	cv::Mat depthMatRaw;
	cv::Mat depthMat_m;
	cv::Mat depthMat_mm;
	cv::Mat depthGreyMat;

	cv::Mat pixelXPosMat;
	cv::Mat pixelYPosMat;

	rs2_intrinsics depthIntrinsics;
	rs2_intrinsics clrIntrinsics;
	rs2_extrinsics sensorExtrinsics;

private:
	void computePixelDistortionaMap();

	rs2::pipeline pipe;
	rs2::frameset newFrameset;
	float depthUnits_m;

};

#endif /* SRC_REALSENSEHELPER_H_ */
