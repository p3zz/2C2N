cmake_minimum_required(VERSION 3.5)

project(NeuralNetwork)


add_executable(main src/main.c)

add_subdirectory(lib)
target_link_libraries(main m lib)
target_include_directories(main PRIVATE lib)

add_subdirectory(test)