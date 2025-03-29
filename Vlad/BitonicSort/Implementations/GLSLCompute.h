#pragma once
#include <glad.h>
#include <GLFW/glfw3.h>

class GLSLCompute {
	GLFWwindow* window;
	static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
public:
	GLSLCompute();
	~GLSLCompute();

	int* Sort(int* arr);

	static GLuint LoadComputeShader(const char* path);
	static GLuint CreateComputeProgram(const char* shaderPath);

};