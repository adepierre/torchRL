
find_package(Torch QUIET)

if(NOT Torch_FOUND)
message(FATAL_ERROR "LibTorch not found, you can download it from https://pytorch.org/get-started/locally/. Once installed, make sure you update Torch_DIR so cmake can find it")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
