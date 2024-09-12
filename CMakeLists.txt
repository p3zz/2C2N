cmake_minimum_required(VERSION 3.5)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(NeuralNetwork)


# add_executable(main src/main.c)

add_subdirectory(lib)
# target_link_libraries(main m lib)
# target_include_directories(main PRIVATE lib)

# enable_testing()
# add_subdirectory(test)