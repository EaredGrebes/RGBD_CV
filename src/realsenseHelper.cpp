#include "realsenseHelper.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// helper functions
////////////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------
void displaySensorInfo(rs2_intrinsics &intrin){

    std::cout << "model: " << intrin.model << std::endl;
    std::cout << "width: " << intrin.width << std::endl;
    std::cout << "height: " << intrin.height << std::endl;
    std::cout << "ppx: " << intrin.ppx << std::endl;
    std::cout << "ppy: " << intrin.ppy << std::endl;
    std::cout << "fx: " << intrin.fx << std::endl;
    std::cout << "fy: " << intrin.fy << std::endl;

    std::cout << "coeff [0]: " << intrin.coeffs[0] << std::endl;
    std::cout << "coeff [1]: " << intrin.coeffs[1] << std::endl;
    std::cout << "coeff [2]: " << intrin.coeffs[2] << std::endl;
    std::cout << "coeff [3]: " << intrin.coeffs[3] << std::endl;
    std::cout << "coeff [4]: " << intrin.coeffs[4] << std::endl;
}


//--------------------------------------------------------------
void deproject_pixel_to_point_distBronwCon(float &xCorrected, float &yCorrected, const rs2_intrinsics &depthIntrinsics, const float pixel[2])
{

   float x = (pixel[0] - depthIntrinsics.ppx) / depthIntrinsics.fx;
   float y = (pixel[1] - depthIntrinsics.ppy) / depthIntrinsics.fy;

   float xo = x;
   float yo = y;

   // iterative part
   for (int i = 0; i < 10; i++)
   {
       float r2 = x * x + y * y;
       float icdist = (float)1 / (float)(1 + ((depthIntrinsics.coeffs[4] * r2 + depthIntrinsics.coeffs[1])*r2 + depthIntrinsics.coeffs[0])*r2);
       float delta_x = 2 * depthIntrinsics.coeffs[2] * x*y + depthIntrinsics.coeffs[3] * (r2 + 2 * x*x);
       float delta_y = 2 * depthIntrinsics.coeffs[3] * x*y + depthIntrinsics.coeffs[2] * (r2 + 2 * y*y);
       x = (xo - delta_x)*icdist;
       y = (yo - delta_y)*icdist;
   }

   xCorrected = x;
   yCorrected = y;
}

////////////////////////////////////////////////////////////////////////////////
// class methods
////////////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------
realsenseHelper::realsenseHelper(int meshWidth, int meshHeight) {

	widthPixels = meshWidth;
	heightPixels = meshHeight;

    // raw color (r,g,b) as a uchar3
	colorMat.create(cv::Size(meshWidth, meshHeight), CV_8UC3);

    // raw depth data stored as a unsigned 16 bit value
	depthMatRaw.create(cv::Size(meshWidth, meshHeight), CV_16U);

    // depth data converted to m, stored as float.
    depthMat_m.create(cv::Size(meshWidth, meshHeight), CV_32F);

    // depth data converted to mm, stored as int.  
    // why int?  so cuda atomic_min can be used in depth testing
	depthMat_mm.create(cv::Size(meshWidth, meshHeight), CV_32S);

    // convert the depth data to greyscale for visualization
    depthGreyMat.create(cv::Size(meshWidth, meshHeight), CV_8UC1);

 	rs2::context ctx;
	rs2::device_list devices = ctx.query_devices();

    rs2::device selectedDevice;
    rs2::sensor selectedSensor;
    rs2::stream_profile strProfile;

    unsigned int idxDevice = 0;
    selectedDevice = devices[idxDevice];

    // 0 : Stereo Module
    // 1 : RGB Camera
    // 2 : Motion Module
    int idxSensor = 0;
    selectedSensor = get_a_sensor_from_a_device(selectedDevice, idxSensor);
    std::vector<rs2::stream_profile> strmProfiles = selectedSensor.get_stream_profiles();

    //std::cout << "stream profiles for this sensor: " << std::endl;
    //for (int ii = 0; ii < strmProfiles.size(); ii++){
    //	std::cout << strmProfiles[ii].stream_type() << std::endl;
    //}

    // for info
    std::cout << " : " << get_device_name(devices[0]) << std::endl;
    print_device_information(selectedDevice);

    // realsense Start streaming with default recommended configuration
    //pipe.start();
    rs2::config cfg;
    //cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 15);
    cfg.enable_stream(RS2_STREAM_DEPTH, meshWidth, meshHeight, RS2_FORMAT_Z16, 60);
    cfg.enable_stream(RS2_STREAM_COLOR, meshWidth, meshHeight, RS2_FORMAT_BGR8, 60);

    rs2::pipeline_profile pipeProfile = pipe.start(cfg);

    depthIntrinsics = pipeProfile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    clrIntrinsics = pipeProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    sensorExtrinsics = pipeProfile.get_stream(RS2_STREAM_DEPTH).get_extrinsics_to(pipeProfile.get_stream(RS2_STREAM_COLOR));

    // Testing: modify translation to better match observed shadowing from depth/color corelation
    sensorExtrinsics.translation[0] = sensorExtrinsics.translation[0] + 0.007;
    //sensorExtrinsics.translation[1] = sensorExtrinsics.translation[1] - 0.002;
    //sensorExtrinsics.translation[2] = sensorExtrinsics.translation[2] - 0.002;

    std::cout << "\n\n depth sensor \n\n" << std::endl;
    displaySensorInfo(depthIntrinsics);

    std::cout << "\n\n RGB sensor \n\n" << std::endl;
    displaySensorInfo(clrIntrinsics);

    std::cout << "\n\n camera rotation" << std::endl;
    for (int ii = 0; ii < 9; ii++){
        std::cout << sensorExtrinsics.rotation[ii] << std::endl;
    }

    std::cout << "\n\n camera translation" << std::endl;
    for (int ii = 0; ii < 3; ii++){
        std::cout << sensorExtrinsics.translation[ii] << std::endl;
    }

    computePixelDistortionaMap();
}


//--------------------------------------------------------------
void realsenseHelper::computePixelDistortionaMap() {

    // find the corrected X and Y pixel coordinates for the depth camera
    pixelXPosMat = cv::Mat(cv::Size(widthPixels, heightPixels), CV_32F);
    pixelYPosMat = cv::Mat(cv::Size(widthPixels, heightPixels), CV_32F);

    float pixel[2];
    float xCorrected = 0;
    float yCorrected = 0;

    for (unsigned int ii = 0; ii < widthPixels; ii++){
        for (unsigned int jj = 0; jj < heightPixels; jj++){

            // just linearly scale the pixel x,y indexes to position units
            bool linearProj = false;
            if (linearProj) {

                float xScaled = (float(ii) - depthIntrinsics.ppx) / depthIntrinsics.fx;
                float yScaled = (float(jj) - depthIntrinsics.ppy) / depthIntrinsics.fy;

                pixelXPosMat.at<float>(jj,ii) = xScaled;
                pixelYPosMat.at<float>(jj,ii) = yScaled;

            // use the full nonlinear correction
            } else {

                pixel[0] = ii;
                pixel[1] = jj;

                deproject_pixel_to_point_distBronwCon(xCorrected, yCorrected, depthIntrinsics, pixel);

                pixelXPosMat.at<float>(jj,ii) = xCorrected;
                pixelYPosMat.at<float>(jj,ii) = yCorrected;
            }
        }
    }
}


//--------------------------------------------------------------
rs2::pipeline realsenseHelper::get_pipe()
{
	return pipe;
}


//--------------------------------------------------------------
bool realsenseHelper::checkForNewFrame()
{
	bool newFrameAvail = pipe.poll_for_frames(&newFrameset);

	if (newFrameAvail == true){

		rs2::video_frame colorFrame = newFrameset.get_color_frame();
		rs2::depth_frame depthFrame = newFrameset.get_depth_frame();

		colorMat = cv::Mat(cv::Size(widthPixels, heightPixels), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
		depthMatRaw = cv::Mat(cv::Size(widthPixels, heightPixels), CV_16U, (void*)depthFrame.get_data(), cv::Mat::AUTO_STEP);

		depthMatRaw.convertTo(depthMat_m, CV_32F);
		depthMat_m = depthMat_m * depthFrame.get_units();
        cv::Mat tmp1 = depthMat_m * 1000.0f; 
        tmp1.convertTo(depthMat_mm, CV_32S);

        //cv::Mat tmp = depthMat_m/(50*0.001);  // arbitrary scale to get depth to map to 8-bit grey scale
        cv::Mat tmp = depthMat_mm / 10;  // arbitrary scale to get depth to map to 8-bit grey scale
        tmp.convertTo(depthGreyMat, CV_8UC1);
	}
	return newFrameAvail;
}


//--------------------------------------------------------------
std::string realsenseHelper::get_device_name(const rs2::device& dev)
{
    // Each device provides some information on itself, such as name:
    std::string name = "Unknown Device";
    if (dev.supports(RS2_CAMERA_INFO_NAME))
        name = dev.get_info(RS2_CAMERA_INFO_NAME);

    // and the serial number of the device:
    std::string sn = "########";
    if (dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER))
        sn = std::string("#") + dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

    return name + " " + sn;
}


//--------------------------------------------------------------
void realsenseHelper::print_device_information(const rs2::device& dev)
{
    // Each device provides some information on itself
    // The different types of available information are represented using the "RS2_CAMERA_INFO_*" enum

    std::cout << "Device information: " << std::endl;
    //The following code shows how to enumerate all of the RS2_CAMERA_INFO
    //Note that all enum types in the SDK start with the value of zero and end at the "*_COUNT" value
    for (int i = 0; i < static_cast<int>(RS2_CAMERA_INFO_COUNT); i++)
    {
        rs2_camera_info info_type = static_cast<rs2_camera_info>(i);
        //SDK enum types can be streamed to get a string that represents them
        std::cout << info_type << " : " << std::endl;

        //A device might not support all types of RS2_CAMERA_INFO.
        //To prevent throwing exceptions from the "get_info" method we first check if the device supports this type of info
        if (dev.supports(info_type))
            std::cout << dev.get_info(info_type) << std::endl;
        else
            std::cout << "N/A" << std::endl;
    }
}


//--------------------------------------------------------------
rs2::sensor realsenseHelper::get_a_sensor_from_a_device(const rs2::device& dev, int selected_sensor_index)
{
    // A rs2::device is a container of rs2::sensors that have some correlation between them.
    // For example:
    //    * A device where all sensors are on a single board
    //    * A Robot with mounted sensors that share calibration information

    // Given a device, we can query its sensors using:
    std::vector<rs2::sensor> sensors = dev.query_sensors();

    std::cout << "Device consists of " << sensors.size() << " sensors:\n" << std::endl;
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
    {
        std::cout << "  " << index++ << " : " << get_sensor_name(sensor) << std::endl;
    }

    // The second way is using the subscript ("[]") operator:
    if (selected_sensor_index >= sensors.size())
    {
        throw std::out_of_range("Selected sensor index is out of range");
    }

    return  sensors[selected_sensor_index];
}


//--------------------------------------------------------------
std::string realsenseHelper::get_sensor_name(const rs2::sensor& sensor)
{
    // Sensors support additional information, such as a human readable name
    if (sensor.supports(RS2_CAMERA_INFO_NAME))
        return sensor.get_info(RS2_CAMERA_INFO_NAME);
    else
        return "Unknown Sensor";
}


//--------------------------------------------------------------
realsenseHelper::~realsenseHelper() {

}

