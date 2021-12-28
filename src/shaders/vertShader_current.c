#version 400 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec4 ourColor;

uniform mat4 ModelViewProjectionMat;


void main()
{
    gl_Position = ModelViewProjectionMat * vec4(aPos, 1.0);
    ourColor = vec4(aColor.zyx/255, 1.0);
}
