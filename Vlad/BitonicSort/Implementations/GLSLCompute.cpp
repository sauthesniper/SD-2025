#include "GLSLCompute.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"


 GLSLCompute::GLSLCompute(int screenWidth, int screenHeight) {
 	glfwInit();

	this->screenWidth = screenWidth;
 	this->screenHeight = screenHeight;

 	GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
 	const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);

 	window = glfwCreateWindow(mode->width, mode->height, "OpenGL Test", primaryMonitor, NULL);
 	glfwMakeContextCurrent(window);
 	glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);

 	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
 		std::cerr << "Failed to initialize GLAD" << std::endl;
 		return;
 	}


	std::cout << "OpenGL Initialized successfully" << std::endl;
 }
 GLSLCompute::~GLSLCompute() {
 	glfwDestroyWindow(window);
 	glfwTerminate();
 }


int* GLSLCompute::Sort(int* arr, int size) {
 	GLuint ssbo;

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


void GLSLCompute::AnimSort(int* arr, int size) {
	int pow2 = 0;
 	int aux = 1;
 	while(aux < size) {
 		aux *= 2;
 		pow2++;
 	}

 	int width = size;
 	int height = pow2 * (pow2 + 1) / 2 + 1;
	GLuint sortingTexture;
	glGenTextures(1, &sortingTexture);
	glBindTexture(GL_TEXTURE_2D, sortingTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, nullptr);


	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	std::vector<int> zeros(width * height, 0);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_INT, zeros.data());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, 1, GL_RED_INTEGER, GL_INT, arr);

	GLuint computeProgram = CreateComputeProgram("../Shaders/bitonicSortAnim.comp");
	int imageUnitIndex = 0;
	int imageUniformLoc = glGetUniformLocation(computeProgram, "sortingTexture");
	int stageLoc = glGetUniformLocation(computeProgram, "stage");
	int passLoc = glGetUniformLocation(computeProgram, "pass");
	int rowLoc = glGetUniformLocation(computeProgram, "row");

	int row = 0;
	for(int s = 2; s <= size; s *= 2) {
		for(int p = s / 2; p > 0; p /= 2) {
			glUseProgram(computeProgram);
			glUniform1i(stageLoc, s);
			glUniform1i(passLoc, p);
			glUniform1i(rowLoc, row);
			glUniform1i(imageUniformLoc, imageUnitIndex); //program must be active
			glBindImageTexture(imageUnitIndex, sortingTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
			glDispatchCompute((width + 15) / 16, 1, 1);
			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
			row++;
		}
	}

	// std::vector<int> debugData(width * height);
	// glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_INT, debugData.data());
	//
	// for (int y = 0; y < height; y++) {
	// 	for (int x = 0; x < width; x++) {
	// 		std::cout << debugData[y * width + x] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	//
	// SaveTexture(sortingTexture, width, height, "C:/Dalv/tex.png");

	GLuint fragmentShaderProgram = CreateShaderProgram("../Shaders/defaultVertex.vert", "../Shaders/bitonicVisualizer.frag");
	imageUniformLoc = glGetUniformLocation(fragmentShaderProgram, "sortingTexture");
	int timeLoc = glGetUniformLocation(fragmentShaderProgram, "time");
	int widthLoc = glGetUniformLocation(fragmentShaderProgram, "width");
	int screenWidthLoc = glGetUniformLocation(fragmentShaderProgram, "screenWidth");
	int screenHeightLoc = glGetUniformLocation(fragmentShaderProgram, "screenHeight");
	int heightLoc = glGetUniformLocation(fragmentShaderProgram, "height");
	int maxValueLoc = glGetUniformLocation(fragmentShaderProgram, "maxValue");

	// Full-screen quad vertices (two triangles)
	GLfloat quadVertices[] = {
		// positions        // texCoords
		-1.0f,  1.0f,      0.0f, 1.0f,
		-1.0f, -1.0f,      0.0f, 0.0f,
		 1.0f, -1.0f,      1.0f, 0.0f,
		-1.0f,  1.0f,      0.0f, 1.0f,
		 1.0f, -1.0f,      1.0f, 0.0f,
		 1.0f,  1.0f,      1.0f, 1.0f
	};

	// Create VAO and VBO for the quad
	GLuint quadVAO, quadVBO;
	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);

	// Set up the quad's vertex data
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

	// Set up vertex attributes for position and texture coordinates
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	// Unbind VAO and VBO (optional)
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


 	while (!glfwWindowShouldClose(window)) {
 		float currentTime = glfwGetTime();

 		glClear(GL_COLOR_BUFFER_BIT);

 		// Bind Fragment Shader for Visualization
 		glUseProgram(fragmentShaderProgram);
 		glUniform1i(imageUniformLoc, imageUnitIndex); //program must be active
 		glUniform1f(timeLoc, currentTime); //program must be active
 		glUniform1i(widthLoc, width); //program must be active
 		glUniform1i(screenWidthLoc, screenWidth); //program must be active
 		glUniform1i(screenHeightLoc, screenHeight); //program must be active
 		glUniform1i(heightLoc, height); //program must be active
 		glUniform1i(maxValueLoc, width); //program must be active
 		glBindImageTexture(imageUnitIndex, sortingTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32I);

 		// Bind the full-screen quad VAO
 		glBindVertexArray(quadVAO);
 		// Draw the quad
 		glDrawArrays(GL_TRIANGLES, 0, 6);

 		glfwSwapBuffers(window);
 		glfwPollEvents();
 	}
 }

GLuint GLSLCompute::LoadShader(GLenum type, const char* path) {
 	std::ifstream file(path);
 	std::stringstream buffer;
 	buffer << file.rdbuf();
 	std::string shaderCode = buffer.str();
 	const char* source = shaderCode.c_str();

 	GLuint shader = glCreateShader(type);
 	glShaderSource(shader, 1, &source, nullptr);
 	glCompileShader(shader);

 	int success;
 	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
 	if (!success) {
 		char infoLog[512];
 		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
 		std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
 		return 0;
 	}

 	return shader;
 }

GLuint GLSLCompute::CreateComputeProgram(const char* shaderPath) {
	GLuint computeShader = LoadShader(GL_COMPUTE_SHADER, shaderPath);
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

GLuint GLSLCompute::CreateShaderProgram(const char* vertexPath, const char* fragmentPath) {
 	GLuint vertexShader = LoadShader(GL_VERTEX_SHADER, vertexPath);
 	GLuint fragmentShader = LoadShader(GL_FRAGMENT_SHADER, fragmentPath);

 	GLuint program = glCreateProgram();
 	glAttachShader(program, vertexShader);
 	glAttachShader(program, fragmentShader);
 	glLinkProgram(program);

 	int success;
 	glGetProgramiv(program, GL_LINK_STATUS, &success);
 	if (!success) {
 		char infoLog[512];
 		glGetProgramInfoLog(program, 512, nullptr, infoLog);
 		std::cerr << "Shader Program Linking Failed:\n" << infoLog << std::endl;
 	}

 	glDeleteShader(vertexShader);
 	glDeleteShader(fragmentShader);

 	return program;

 }


void GLSLCompute::SaveTexture(GLuint textureID, int width, int height, const char* filename) {
 	// Bind texture
 	glBindTexture(GL_TEXTURE_2D, textureID);

 	// Allocate buffer to store pixels
 	std::vector<int> pixels(width * height);

 	// Read texture data (assuming GL_R32I integer format)
 	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_INT, pixels.data());

 	// Convert integer texture to 8-bit grayscale for saving
 	std::vector<unsigned char> image(width * height);
 	for (size_t i = 0; i < pixels.size(); i++) {
 		image[i] = static_cast<unsigned char>(pixels[i] % 256); // Normalize
 	}

 	// Save as PNG (1 channel, grayscale)
 	if (stbi_write_png(filename, width, height, 1, image.data(), width)) {
 		std::cout << "Texture saved to " << filename << std::endl;
 	} else {
 		std::cerr << "Failed to save texture!" << std::endl;
 	}
 }




void GLSLCompute::FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
