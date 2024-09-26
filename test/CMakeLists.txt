include(CTest)

# include unity
add_library(unity STATIC
    ../Unity/src/unity.c
)

target_include_directories(unity PUBLIC
    ../Unity/src
)

# include cnn library for testing
set(CNN_FRAMEWORK_SRC_FILES
    ../src/utils.c
    ../src/common.c
    ../src/layer.c
)

add_library(c_cnn_framework_test STATIC "${CNN_FRAMEWORK_SRC_FILES}")
# include the custom header before the default header so we can overwrite the types.h content
target_include_directories(c_cnn_framework_test PUBLIC ../include env)

add_executable(test_common
    test_common.c
)

target_link_libraries(test_common
    c_cnn_framework_test
    unity
    m
)

add_test(test_common test_common)

add_executable(test_layer
    test_layer.c
)

target_link_libraries(test_layer
    c_cnn_framework_test
    unity
    m
)

add_test(test_layer test_layer)
