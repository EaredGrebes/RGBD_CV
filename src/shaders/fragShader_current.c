#version 400 core

in vec4 ourColor;

out vec4 FragColor;

void main()
{
    FragColor = ourColor;
    //FragColor = vec4(0.5, 0.0, 0.5, 1.0);   //testing
    
	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	if (dot(circCoord, circCoord) > 1.0) {
		discard;
	}
}
