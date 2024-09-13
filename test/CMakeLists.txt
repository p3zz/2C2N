include(CTest)

add_library(unity STATIC
    ../Unity/src/unity.c
)

target_include_directories(unity PUBLIC
    ../Unity/src
)

add_executable(test_common
    test_common.c
)

target_link_libraries(test_common
    c_cnn_framework
    unity
    m
)


target_include_directories(test_common PRIVATE ../lib ../Unity)

add_test(test_common test_common)

add_executable(test_layer
    test_layer.c
)

target_link_libraries(test_layer
    c_cnn_framework
    unity
    m
)


target_include_directories(test_layer PRIVATE ../lib ../Unity)

add_test(test_layer test_layer)
