cmake_minimum_required(VERSION 3.5)

project(NeuralNetwork)

set(LIB_FILES
    src/utils.h
    src/utils.c
    src/neuron.c
    src/neuron.h
    src/layer.c
    src/layer.h
    src/network.h
    src/network.c
)

add_library(lib ${LIB_FILES})
add_executable(main src/main.c)

target_link_libraries(main m lib)
