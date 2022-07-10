project(torchrl)

set(hdr_files
    include/torchrl/algorithms/ppo/PPO.hpp
    include/torchrl/algorithms/ppo/PPOArgs.hpp
    
    include/torchrl/envs/AbstractEnv.hpp
    include/torchrl/envs/RunningMeanStd.hpp
    include/torchrl/envs/VectorizedEnv.hpp
    
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
    
    src/rl/MLP.cpp
    src/rl/NormalDistribution.cpp
    src/rl/Policy.cpp
    src/rl/RolloutBuffer.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${hdr_files} ${src_files}
)

add_library(${PROJECT_NAME} STATIC ${hdr_files} ${src_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")