#include "GLSLCompute.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

 GLSLCompute::GLSLCompute() {
 	glfwInit();

 	window = glfwCreateWindow(800, 600, "OpenGL Test", NULL, NULL);
 	glfwMakeContextCurrent(window);
 	glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);

 	assert(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress));

	std::cout << "OpenGL Initialized successfully" << std::endl;
 }
 GLSLCompute::~GLSLCompute() {
 	glfwDestroyWindow(window);
 	glfwTerminate();
 }


int* GLSLCompute::Sort(int* arr) {
 	GLuint ssbo;

 	int size = sizeof(arr) / sizeof(arr[0]);

 	// Generate and bind SSBO
 	glGenBuffers(1, &ssbo);
 	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
 	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(arr), arr, GL_DYNAMIC_COPY);
 	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo); // Binding it to binding point 0
 	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

 	GLuint computeProgram = CreateComputeProgram("../Shaders/test.comp");

 	// Use the compute shader program
 	glUseProgram(computeProgram);
	int workGroupSize = 256;
	glDispatchCompute((size + workGroupSize - 1) / workGroupSize, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);  // Ensure sync before reading back

 	// Retrieve data from buffer
 	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
 	int* sortedArr = (int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
 	if (sortedArr) {
 		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 	}
 	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

 	glDeleteProgram(computeProgram);

 	return sortedArr;
 }



GLuint GLSLCompute::LoadComputeShader(const char* path) {
	std::ifstream fin(path);
	assert(fin.is_open());

	std::stringstream buffer;
	buffer << fin.rdbuf();
	fin.close();

	std::string shaderCode = buffer.str();
	const char* source = shaderCode.c_str();

	std::cout << "Loaded Compute Shader" << std::endl; // Debug: print shader

	GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	// Check for compilation errors
	int success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cerr << "Compute Shader Compilation Failed:\n" << infoLog << std::endl;
		return 0;
	}


	return shader;
}

GLuint GLSLCompute::CreateComputeProgram(const char* shaderPath) {
	GLuint computeShader = LoadComputeShader(shaderPath);
	GLuint program = glCreateProgram();
	glAttachShader(program, computeShader);
	glLinkProgram(program);

	// Check for linking errors
	int success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cerr << "Compute Program Linking Failed:\n" << infoLog << std::endl;
	}

	glDeleteShader(computeShader);
	return program;
}

void GLSLCompute::FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
