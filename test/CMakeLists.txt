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
    2c2n
    unity
    m
)

add_test(test_common test_common)

add_executable(test_layer
    test_layer.c
)

target_link_libraries(test_layer
    2c2n
    unity
    m
)

add_test(test_layer test_layer)

add_executable(test_matrix
    test_matrix.c
)

target_link_libraries(test_matrix
    2c2n
    unity
    m
)

add_test(test_matrix test_matrix)
