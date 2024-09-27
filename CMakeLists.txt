cmake_minimum_required(VERSION 3.5)

project(NeuralNetwork)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic -Werror")

set(LIB_SOURCES
    src/utils.c
    src/common.c
    src/layer.c
)

add_library(c_cnn_framework STATIC "${LIB_SOURCES}")
target_include_directories(c_cnn_framework PUBLIC include)

enable_testing()
add_subdirectory(test)
