## CCNN framework
### Test
2 tests are available in test/ folder:
- test_common
- test_layer

### Tools
Several tools are available in tools/ folder:
- build.sh: build all targets (CMake)
- clear.sh: remove the build folder
- test.sh: run tests in verbose mode (CTest)
- format.sh: code formatting (Clang Format)
- heap-check.sh: run memory leaks check over tests (Valgrind)

https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view

## Convolutional layer
![Convolutional layer - Forwarding](./assets/convolutional_layer_forwarding.jpg)
![Dense layer - Forwarding](./assets/dense_layer_forwarding.jpg)
![Average pooling layer - Forwarding](./assets/avg_pooling_layer_forwarding.jpg)
![Max pooling layer - Forwarding](./assets/max_pooling_layer_forwarding.jpg)
![Max pooling layer - Forwarding (example)](./assets/max_pooling_layer_forwarding_example.jpg)