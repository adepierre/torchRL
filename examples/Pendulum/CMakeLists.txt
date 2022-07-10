project(Pendulum)

set(hdr_files
    include/Pendulum/PendulumEnv.hpp
)

set(src_files
    src/PendulumEnv.cpp
    
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${hdr_files} ${src_files})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PRIVATE torchrl)

# We want all the executables for the examples to be at the same place
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../bin"
)

#The following code block is suggested to be used on Windows.
#According to https://github.com/pytorch/pytorch/issues/25457,
#the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)