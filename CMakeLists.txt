cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(TorchRL)

#############################################
################## DEPENDS ##################
#############################################
set(FETCHCONTENT_QUIET FALSE)
include(FetchContent)

# Torch
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/torch.cmake")

# OpenGL
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/opengl.cmake")

# GLFW
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/glfw.cmake")

# Glad
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/glad.cmake")

# Imgui
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/imgui.cmake")

# Implot
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/implot.cmake")

#############################################
################## CONTENT ##################
#############################################
add_subdirectory(torchrl)
add_subdirectory(examples/Pendulum)
add_subdirectory(examples/MountainCar)
