#! /bin/bash

valgrind --leak-check=full build/test/test_layer && \
valgrind --leak-check=full build/test/test_common