add_executable(test_network
    test_network.c
)

target_link_libraries(test_network
    lib
    Unity
)

add_test(test_network test_network)