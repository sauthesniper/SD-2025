cmake_minimum_required(VERSION 3.30)
project(BitonicSort LANGUAGES C CXX CUDA)


# Include directories for GLFW, CUDA, and GLAD
include_directories(${CMAKE_SOURCE_DIR}/libs/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)  # Ensure glad.c is available
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")

# Link directories for GLFW and other libraries
link_directories(${CMAKE_SOURCE_DIR}/libs/lib-vc2022)
link_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64)

# Add GLAD to the project as a STATIC library
add_library(glad STATIC src/glad.c)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)

# Create the executable
add_executable(BitonicSort main.cpp BitonicKernel.cu
        Implementations/GLSLCompute.cpp
        Implementations/GLSLCompute.h
        Implementations/CudaCompute.h
        include/stb_image_write.h
)

# Enable CUDA for the project
set_target_properties(BitonicSort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link GLAD and other libraries
target_link_libraries(BitonicSort glad glfw3.lib opengl32.lib cudadevrt.lib cudart_static.lib)

# Link Windows libraries (optional)
target_link_libraries(BitonicSort kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib)
