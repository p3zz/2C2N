cmake_minimum_required(VERSION 3.5)

project(NeuralNetwork)

set(LIB_SOURCES
    src/common.c
    src/layer.c
    src/utils.c
)

add_library(lib "${LIB_SOURCES}")
target_include_directories(lib PUBLIC include)

enable_testing()
add_subdirectory(test)
