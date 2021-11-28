#ifndef SHADERHELPER_H
#define SHADERHELPER_H

#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
  
GLuint createShaderProgram();
GLuint createShaderProgram(const char* vertexPath, const char* fragmentPath);

#endif