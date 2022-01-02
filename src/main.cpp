// C++ stuff
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <thread>
#include <mutex>
#include <boost/filesystem.hpp>

// opencv
#include <opencv2/opencv.hpp>
//#include <opencv2/cudaarithm.hpp>
#include "opencv2/core/cuda.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/hdf.hpp>
#include <opencv2/hdf/hdf5.hpp>

// openGL
#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// my own classes
#include "shaderProgramGenerator.h"
#include "realsenseHelper.h"
#include "glUserInputHelper.h"
#include "cuda_processing_RGBD.h"


////////////////////////////////////////////////////////////////////////////////
// constants, paremeters, and objects
////////////////////////////////////////////////////////////////////////////////

unsigned int windowW = 1700, windowH = 900;

const unsigned int meshWidth    = 848;
const unsigned int meshHeight   = 480;
const unsigned int Npoints = meshWidth * meshHeight;

// OpenGL vertex buffers
GLuint pointVertexBuffer;
struct cudaGraphicsResource *cuda_3dPointVB_resource; // handles OpenGL-CUDA exchange
struct cudaGraphicsResource *cuda_colorVB_resource;
GLuint shaderProgram;

// timing
float deltaTime_sec = 0.0f; // time between current frame and last frame
float lastFrameTime_sec = 0.0f;
float frameRateFilt = 0.0f;
float frameRateTimeConst = 0.95f;

// openCV - data for display and saving
cv::Mat colorInDepthMat(cv::Size(meshWidth, meshHeight), CV_8UC3);
cv::Mat depthMat8L_mm(cv::Size(meshWidth, meshHeight), CV_8U);
cv::Mat depthMat8U_mm(cv::Size(meshWidth, meshHeight), CV_8U);
cv::Mat posMat_m(cv::Size(meshWidth, meshHeight), CV_32FC3);
cv::Mat edgeMaskMat(cv::Size(meshWidth, meshHeight), CV_8U);

// custom objects
GLFWwindow* window;

realsenseHelper realsenseHelper(meshWidth, meshHeight);

glUserInputHelper usrInput;

cuda_processing_RGBD cudaRGBD(meshWidth, 
                              meshHeight,
                              realsenseHelper.sensorExtrinsics.rotation, 
                              realsenseHelper.sensorExtrinsics.translation);  

// video capture, which uses multi-threading
bool newDataFlag1 = false;
bool newDataFlag2 = false;
bool newDataFlag3 = false;
bool startRecord = false;
bool stopRecord = false;
std::mutex mutex;
//int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
//int fourcc = cv::VideoWriter::fourcc('F', 'F', 'V', '1');
//int fourcc = cv::VideoWriter::fourcc('F','M','P','4');
//int fourcc = cv::VideoWriter::fourcc('p', 'n', 'g', ' ');
// last argument is if it's color or not
int vidFps = 59;
cv::VideoWriter videoCaptureRGB("data/videoCaptureTest1.avi", cv::VideoWriter::fourcc('M','J','P','G'), vidFps, cv::Size(meshWidth, meshHeight), true);
cv::VideoWriter videoCaptureDL("data/videoCaptureTest2.avi", cv::VideoWriter::fourcc('F','F','V','1'), vidFps, cv::Size(meshWidth, meshHeight), false);
cv::VideoWriter videoCaptureDU("data/videoCaptureTest3.avi", cv::VideoWriter::fourcc('F','F','V','1'), vidFps, cv::Size(meshWidth, meshHeight), false);

////////////////////////////////////////////////////////////////////////////////
// forward declarations
////////////////////////////////////////////////////////////////////////////////

bool initGL(GLuint &shaderProg, GLFWwindow* &window);

void threadWriteVideo(cv::VideoWriter &writer, 
                      bool &newDataFlag, 
                      bool &startRecord, 
                      bool &stopRecord, 
                      std::mutex &mutex, 
                      cv::Mat &matIn);

void runMainLoop(GLFWwindow* window);

// static function wrappers for glfw callbacks
void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    usrInput.framebuffer_size_callback(window, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos){
    usrInput.mouse_callback(window, xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    usrInput.scroll_callback(window, xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    usrInput.mouse_button_callback(window, button, action, mods);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif  

    // some opencv info
    std::cout << cv::getBuildInformation() << std::endl;

    // upload pre-computed pixel x/y locations to the GPU
    cudaRGBD.pixelXPosMat.upload(realsenseHelper.pixelXPosMat);
    cudaRGBD.pixelYPosMat.upload(realsenseHelper.pixelYPosMat);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(shaderProgram, window);

    findCudaDevice(argc, (const char **)argv);

    // save calibration data in .h5
    cv::String filename = "data/calibration.h5";
    if (boost::filesystem::exists(filename)){
        std::cout << filename << " already exists" << std::endl;
    } else {
        std::cout << "writing " << filename << std::endl;
        cv::Ptr<cv::hdf::HDF5> h5CalFile = cv::hdf::open(filename);
        h5CalFile->dswrite(realsenseHelper.pixelXPosMat, "pixelXPosMat");
        h5CalFile->dswrite(realsenseHelper.pixelYPosMat, "pixelYPosMat");
        h5CalFile->close();
    }

    // start rendering mainloop
    runMainLoop(window);

    // glfw: terminate, clearing all previously allocated GLFW resources
    glfwTerminate();
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
// runMainLoop
////////////////////////////////////////////////////////////////////////////////
void runMainLoop(GLFWwindow* window)
{

    float vertices[Npoints*3] = {};
    GLbyte colors[Npoints*3] = {};

    unsigned int points_VBO, colors_VBO, VAO;

    glGenVertexArrays(1, &VAO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glGenBuffers(1, &points_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, points_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, points_VBO); 
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); 

    glGenBuffers(1, &colors_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, colors_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, colors_VBO); 
    glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_FALSE, 3 * sizeof(GLbyte), (void*)0); 

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0); 

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_3dPointVB_resource, points_VBO, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_colorVB_resource, colors_VBO, cudaGraphicsMapFlagsWriteDiscard));

    // multi-threading for the video capture
    std::thread videoWriteThread1(threadWriteVideo, 
                                  std::ref(videoCaptureRGB), 
                                  std::ref(newDataFlag1), 
                                  std::ref(startRecord),
                                  std::ref(stopRecord),
                                  std::ref(mutex), 
                                  std::ref(colorInDepthMat));

    std::thread videoWriteThread2(threadWriteVideo, 
                                  std::ref(videoCaptureDL), 
                                  std::ref(newDataFlag2), 
                                  std::ref(startRecord),
                                  std::ref(stopRecord),                                  
                                  std::ref(mutex), 
                                  std::ref(depthMat8L_mm));

    std::thread videoWriteThread3(threadWriteVideo, 
                                  std::ref(videoCaptureDU), 
                                  std::ref(newDataFlag3), 
                                  std::ref(startRecord),
                                  std::ref(stopRecord),                                  
                                  std::ref(mutex), 
                                  std::ref(depthMat8U_mm));

    // The main render loop
    while (!glfwWindowShouldClose(window)) {

        // key inputs
        usrInput.processInput(window, deltaTime_sec);


        // get simulation time
        float time_sec = static_cast< float >( glfwGetTime() );

        // process sensor input
        if (realsenseHelper.checkForNewFrame()) {

            // upload new realsense images to GPU
            cudaRGBD.depthMat_mm.upload(realsenseHelper.depthMat_mm);
            cudaRGBD.colorMat.upload(realsenseHelper.colorMat);

            // run CUDA kernel to generate vertex positions
            //std::cout << "realsense new frame \n" << std::endl;

            // cuda processing
            size_t num_bytes;
            float3 *vertexPointsPtr;
            uchar3 *colorPointsPtr;

            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_3dPointVB_resource, 0));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vertexPointsPtr, &num_bytes, cuda_3dPointVB_resource));

            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_colorVB_resource, 0));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&colorPointsPtr, &num_bytes, cuda_colorVB_resource));

            // run a bunch of low level processing on cuda, but the main outputs are:
            // * sync color data to location of depth pixels
            // * color and vertex aray data for openGL
            cudaRGBD.runCudaProcessing(vertexPointsPtr, colorPointsPtr);

            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_3dPointVB_resource, 0));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_colorVB_resource, 0));

            // calculate frame rate, display to title of window
            deltaTime_sec = time_sec - lastFrameTime_sec;
            lastFrameTime_sec = time_sec;
            frameRateFilt = (frameRateTimeConst) * frameRateFilt + (1-frameRateTimeConst) * (1.0f/deltaTime_sec);
            std::string str = "3D Image, fps: " + std::to_string(frameRateFilt);
            glfwSetWindowTitle(window, str.c_str());
            
            // wait for all the threads to finish their video saving process
            if (usrInput.startRecord) {
                if ((newDataFlag1 == false) && (newDataFlag2 == false) && (newDataFlag3 == false)) {

                    // start video frame capture threads, lock the shared memory sources
                    mutex.lock();

                    // check to see if recording should start or stop
                    startRecord = usrInput.startRecord;
                    stopRecord = usrInput.stopRecord;

                    // download images from gpu to cpu
                    cudaRGBD.colorInDepthMat.download(colorInDepthMat);
                    cudaRGBD.depthMat8L_mm.download(depthMat8L_mm);
                    cudaRGBD.depthMat8U_mm.download(depthMat8U_mm);

                    newDataFlag1 = true;
                    newDataFlag2 = true;
                    newDataFlag3 = true;

                    mutex.unlock();
                }
            }
        } // end of new realSense input

        // model/view/perspective matrix
        glm::mat4 ModelMat(1.0);
        glm::mat4 ViewMat(1.0);
        glm::mat4 ProjectionMat(1.0);
        
        //angle = 2.0 * time;
        //ViewMat = glm::rotate(ViewMat, angle, glm::vec3(1.0f, 0.0f, 1.0f));
        ViewMat = glm::lookAt(usrInput.cameraPos, usrInput.cameraPos + usrInput.cameraFront, usrInput.cameraUp);

        ProjectionMat = glm::perspective(glm::radians(usrInput.fov), (float)windowW / (float)windowH, 0.1f, 100.0f);

        glm::mat4 ModelViewProjection =  ProjectionMat * ViewMat * ModelMat;

        // pass matrix to GLSL
        GLint ModelViewProjectionMat = glGetUniformLocation(shaderProgram, "ModelViewProjectionMat" );
        glUniformMatrix4fv(ModelViewProjectionMat, 1, GL_FALSE, glm::value_ptr(ModelViewProjection));

        // wipe the drawing surface clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // pick the shader program
        glUseProgram(shaderProgram);

        // this VAO stores the vertex location VBO and fragment color VBO
        glBindVertexArray(VAO);

        // draw each vertex as a point
        glPointSize(3);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, Npoints);

        // put the stuff we've been drawing onto the display
        glfwSwapBuffers(window);

        // update other events like input handling 
        glfwPollEvents();

        bool debug = false;
        if (debug) {

            // use openCV to display additional useful images of the data
            cv::Mat tmp1;
            cudaRGBD.edgeMaskMat.download(tmp1);
            tmp1 *= 255;
            tmp1.convertTo(edgeMaskMat, CV_8U);

            //cv::imshow("raw color image", realsenseHelper.colorMat);
            //cv::imshow("raw depth image", realsenseHelper.depthGreyMat);
            //cv::imshow("color in Depth image", colorInDepthMat);
            cv::imshow("edge mask", edgeMaskMat);

            // for some reason opecv needs this
            int key = cv::waitKey(1);
            if (key == 'q') {
                std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
                break;
            }
        }
    }

    // de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &points_VBO);
    glDeleteBuffers(1, &colors_VBO);
    glDeleteProgram(shaderProgram);

    // stop the video recording, and finish the threads
    mutex.lock();
    stopRecord = true;
    mutex.unlock();

    videoWriteThread1.join();
    videoWriteThread2.join();
    videoWriteThread3.join();

    videoCaptureRGB.release();
    videoCaptureDL.release();
    videoCaptureDU.release();
    cv::destroyAllWindows();
}


////////////////////////////////////////////////////////////////////////////////
// Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(GLuint &shaderProgram, GLFWwindow* &window)
{
    // start GL context and O/S window using the GLFW helper library
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        return 1;
    } 

    // uncomment these lines if on Apple OS X
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(meshWidth, meshHeight, "3D realsense display", NULL, NULL);

    if (!window) {
        fprintf(stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // start GLEW extension handler
    glewExperimental = GL_TRUE;
    glewInit();

    // get version info
    const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString(GL_VERSION);   // version as a string
    printf("Renderer: %s\n", renderer);
    printf("OpenGL version supported %s\n", version);

    // tell GL to only draw onto a pixel if the shape is closer to the viewer
    glEnable(GL_DEPTH_TEST); // enable depth-testing
    glDepthFunc(GL_LESS);    // depth-testing interprets a smaller value as "closer"

    // set the viewing perspective for the render
    glViewport(0, 0, windowW, windowH);

    // self adjusts gl coordinates to re-sized window
    //glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);  

    // create a shader program object
    //GLuint shaderProgram = createShaderProgram();
    shaderProgram = createShaderProgram("src/shaders/vertShader_current.c", "src/shaders/fragShader_current.c");

    return true;
}



////////////////////////////////////////////////////////////////////////////////
// Threaded function uses a VideoWriter object opened in the main thread
////////////////////////////////////////////////////////////////////////////////
void threadWriteVideo(cv::VideoWriter &writer, 
                      bool &newDataFlag, 
                      bool &startRecord, 
                      bool &stopRecord, 
                      std::mutex &mutex, 
                      cv::Mat &matIn)
{

    while (true) {
        if ((true == newDataFlag) && (true == startRecord)){

            writer.write(matIn);

            mutex.lock();
            newDataFlag = false; 
            mutex.unlock();
        }

        // check to see if thread should finish
        if (true == stopRecord){
            return;
        }
    }
}
