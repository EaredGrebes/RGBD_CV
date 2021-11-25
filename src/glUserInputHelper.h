#ifndef SRC_GLUSERINPUTHELPER_H_
#define SRC_GLUSERINPUTHELPER_H_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

// openGL
#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class glUserInputHelper {
public:
	glUserInputHelper();
	virtual ~glUserInputHelper();

	void processInput(GLFWwindow *window, float deltaTime);
	void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

	// camera
	glm::vec3 cameraPos;
	glm::vec3 cameraFront;
	glm::vec3 cameraUp;

	bool firstMouse;
	bool leftButtonPressed;
	bool rightButtonPressed;
	float yaw;   // yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
	float pitch;
	float lastX;
	float lastY;
	float fov;

private:


};

#endif /* SRC_GLUSERINPUTHELPER_H_ */
