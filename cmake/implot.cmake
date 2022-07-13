FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot
  GIT_TAG v0.13
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(glfw imgui implot)

add_library(implot STATIC 
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot PUBLIC imgui)