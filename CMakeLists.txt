cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(PytorchPPO)

set(hdr_files
    include/torchrl/algorithms/ppo/PPO.hpp
    include/torchrl/algorithms/ppo/PPOArgs.hpp
    
    include/torchrl/envs/AbstractEnv.hpp
    include/torchrl/envs/RunningMeanStd.hpp
    include/torchrl/envs/VectorizedEnv.hpp
    
    include/torchrl/envs/impl/MountainCarContinuous.hpp
    include/torchrl/envs/impl/PendulumEnv.hpp
    
    include/torchrl/rl/Args.hpp
    include/torchrl/rl/MLP.hpp
    include/torchrl/rl/NormalDistribution.hpp
    include/torchrl/rl/Policy.hpp
    include/torchrl/rl/RolloutBuffer.hpp
)

set(src_files
    src/algorithms/ppo/PPO.cpp
    
    src/envs/AbstractEnv.cpp
    src/envs/RunningMeanStd.cpp
    src/envs/VectorizedEnv.cpp
    
    src/envs/impl/MountainCarContinuous.cpp
    src/envs/impl/PendulumEnv.cpp
    
    src/rl/MLP.cpp
    src/rl/NormalDistribution.cpp
    src/rl/Policy.cpp
    src/rl/RolloutBuffer.cpp
    
    src/main.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${hdr_files} ${src_files}
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${hdr_files} ${src_files})
target_include_directories(${PROJECT_NAME} PUBLIC include)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

#The following code block is suggested to be used on Windows.
#According to https://github.com/pytorch/pytorch/issues/25457,
#the DLLs need to be copied to avoid memory errors.
# if (MSVC)
  # file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  # add_custom_command(TARGET ${PROJECT_NAME}
                     # POST_BUILD
                     # COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     # ${TORCH_DLLS}
                     # $<TARGET_FILE_DIR:${PROJECT_NAME}>)
# endif (MSVC)