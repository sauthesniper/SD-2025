#pragma once
#include <glad.h>
#include <GLFW/glfw3.h>

class GLSLCompute {
	GLFWwindow* window;
	int screenWidth, screenHeight;

	static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
public:
	GLSLCompute(int screenWidth = 1024, int screenHeight = 800);
	~GLSLCompute();

	int* Sort(int* arr, int size);
	void AnimSort(int* arr, int size);

	static GLuint LoadShader(GLenum type, const char* path);
	static GLuint CreateComputeProgram(const char* shaderPath);
	static GLuint CreateShaderProgram(const char* vertexPath, const char* fragmentPath);
	static void SaveTexture(GLuint textureID, int width, int height, const char* filename);
};