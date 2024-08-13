include(CTest)

add_library(unity STATIC
    ../Unity/src/unity.c
)

target_include_directories(unity PUBLIC
    ../Unity/src
)

add_executable(test_network
    test_network.c
)

target_link_libraries(test_network
    lib
    unity
    m
)

target_include_directories(test_network PRIVATE ../lib ../Unity)

add_test(test_network test_network)
