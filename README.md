# RGBD_CV

The goal of this project was to learn about processing and visualizing color and depth data in real-time using CUDA and C++.  The sensor being used is an Intel Realsense D455.  Disclaimer: this code is not a prototype for production, it is exploratory, meant for learning, and changing constantly.  

The algorithm uses the GPU to:
- kalman filter the depth measurements (the depth sensor has some interesting wavy distortion pattern)
- align the RGB color pixels to the depth estimates.  This has to be done because the two sensors are at different locations and have different lens characteristics.  (this code was mostly copied from the Intel Realsense API, but then implemented in CUDA)
- visualize the data as an interactive point cloud 

In addition, several low level computer vision routines are implemented in CUDA (learning by doing!)
- kernel smoothing
-  edge detection
-  surface normal computation for the depth data 

The last cool thing to mention is an image warping routine, which applies a rotation and translation to the whole RGBD data set.  This will generate new RGB and depth images taking into account the perspective change and elementary shading. 

NOTE: this code has only been developed and tested on linux.
Software requirements:
- CUDA NVCC compiler release 10.1
- OpenGL 4.6.0
- OpenCV with GPU libraries compiled

Beyond code snippets copied from the following sources, everything here is original.

- Intel Realsense C++ API: https://github.com/IntelRealSense/librealsense
- LearnOpenGL: https://github.com/JoeyDeVries/LearnOpenGL
- Fast 3x3 SVD on the GPU (I included the entire file in my src): https://github.com/ericjang/svd3


