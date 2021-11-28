#version 400 core

in vec4 ourColor;

out vec4 FragColor;

void main()
{
    FragColor = ourColor;
    //FragColor = vec4(0.5, 0.0, 0.5, 1.0);
}
