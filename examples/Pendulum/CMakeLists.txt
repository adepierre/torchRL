project(Pendulum)

set(hdr_files
    include/Pendulum/PendulumEnv.hpp
)

set(src_files
    src/PendulumEnv.cpp
    
    src/main.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${hdr_files} ${src_files}
)

add_executable(${PROJECT_NAME} ${hdr_files} ${src_files})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PRIVATE torchrl)
# C++17 is only used to get std::filesystem for plotting data
# generation, you can remove it if you don't use it
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


#The following code block is suggested to be used on Windows.
#According to https://github.com/pytorch/pytorch/issues/25457,
#the DLLs need to be copied to avoid memory errors.
if (MSVC)
    # We want all the executables for the examples to be at the same place
    # to avoid copying the dll multiple times
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../bin"
    )
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif (MSVC)