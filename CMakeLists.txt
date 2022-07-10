cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TorchRL)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_subdirectory(torchrl)
add_subdirectory(examples/Pendulum)
add_subdirectory(examples/MountainCar)
