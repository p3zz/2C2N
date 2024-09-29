cmake_minimum_required(VERSION 3.5)

project(2C2N)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic -Werror")

set(LIB_SOURCES
    src/utils.c
    src/common.c
    src/layer.c
    src/matrix.c
)

add_library(2c2n STATIC "${LIB_SOURCES}")
target_include_directories(2c2n PUBLIC include)

enable_testing()
add_subdirectory(test)
