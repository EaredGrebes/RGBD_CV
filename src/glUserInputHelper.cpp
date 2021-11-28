#include "glUserInputHelper.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

// openGL
#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


////////////////////////////////////////////////////////////////////////////////
// class methods
////////////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------
// constructor
glUserInputHelper::glUserInputHelper() {

    cameraPos   = glm::vec3(0.0f, 0.0f, 3.0f);
    cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

    firstMouse = true;
    leftButtonPressed = false;
    rightButtonPressed = false;
    yaw   = -90.0f;   // yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
    pitch =  0.0f;
    lastX =  800.0f / 2.0;
    lastY =  600.0 / 2.0;
    fov   =  45.0f;

}


// ---------------------------------------------------------------------------------------------------------
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void glUserInputHelper::processInput(GLFWwindow *window, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 3.0 * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}


// ---------------------------------------------------------------------------------------------
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void glUserInputHelper::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    
    //glViewport(0, 0, width, height);
}


// -------------------------------------------------------
// glfw: whenever the mouse moves, this callback is called
void glUserInputHelper::mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xDelta = xpos - lastX;
    float yDelta = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    // use the mouse for rotation
    if (leftButtonPressed) {

        float sensitivity = 0.3f; // change this value to your liking
        xDelta *= sensitivity;
        yDelta *= sensitivity;

        yaw += xDelta;
        pitch += yDelta;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);   
    } 

    // use the mouse for translation
    if (rightButtonPressed) {

        float cameraSpeed = 0.03;
        cameraPos += -glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * xDelta;
        cameraPos += -glm::cross(glm::normalize(glm::cross(cameraFront, cameraUp)), cameraFront) * cameraSpeed * yDelta;
    }

}


// ----------------------------------------------------------------------
// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void glUserInputHelper::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}


// ----------------------------------------------------------------------
// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void glUserInputHelper::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        leftButtonPressed = true;
        firstMouse = true;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        leftButtonPressed = false;
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        rightButtonPressed = true;
        firstMouse = true;
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        rightButtonPressed = false;
    }
        
}


//--------------------------------------------------------------
glUserInputHelper::~glUserInputHelper() {

}

