set(LIB_FILES
    utils.h
    utils.c
    neuron.c
    neuron.h
    layer.c
    layer.h
    network.h
    network.c
    common.h
    common.c
)

add_library(lib ${LIB_FILES})
