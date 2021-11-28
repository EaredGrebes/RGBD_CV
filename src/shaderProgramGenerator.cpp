// code sample frome:
// https://learnopengl.com/code_viewer_gh.php?code=includes/learnopengl/shader_s.h

#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include "shaderProgramGenerator.h"


//---------------------------------------------------------------------------------------
void checkShaderErrors(GLuint shader, std::string errMsg)
{
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << errMsg << infoLog << std::endl;
    }
}


//---------------------------------------------------------------------------------------
GLuint compileAndLink(std::string vertexCode, std::string fragmentCode)
{
    const char* vShaderCode = vertexCode.c_str();
    const char * fShaderCode = fragmentCode.c_str();

    // create shader objects
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vShaderCode, NULL);
    glCompileShader(vertexShader);
    checkShaderErrors(vertexShader, "Error in vertex shader compilation: ");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
    glCompileShader(fragmentShader);  
    checkShaderErrors(fragmentShader, "Error in fragment shader compilation: ");

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, fragmentShader);
    glAttachShader(shaderProgram, vertexShader);
    glLinkProgram(shaderProgram);

    // check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "Error linking shader program: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}


//---------------------------------------------------------------------------------------
GLuint createShaderProgram()
{
    // GLSL program
    std::string vertexCode =
    "#version 400 core\n"
    "layout (location = 0) in vec3 position;"
    "out vec4 color;"
    "uniform mat4 ModelViewProjectionMat;"
    "void main() {"
    "  gl_Position = ModelViewProjectionMat * vec4(position.x, position.y, position.z, 1.0);"
    "  color = vec4(0.5, 0.2, 0.2, 1.0);"
    "}";

    std::string fragmentCode =
    "#version 400 core\n"
    "in vec4 color;"
    "out vec4 fragColor;"
    "void main() {"
    "  fragColor = color;"
    "}";

    return compileAndLink(vertexCode, fragmentCode);
}


//---------------------------------------------------------------------------------------
GLuint createShaderProgram(const char* vertexPath, const char* fragmentPath)
{
    // 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);

    try {
        // open files
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;
        // read file's buffer contents into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        // close file handlers
        vShaderFile.close();
        fShaderFile.close();
        // convert stream into string
        vertexCode   = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    }
    catch (std::ifstream::failure& e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }

    return compileAndLink(vertexCode, fragmentCode);
}



