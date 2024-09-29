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
*Convolutional layer forwarding - input (5x4x4), 2 kernels (3x3x3), 2 biases (3x2), padding 0, stride 1*
![Convolutional layer - Back-propagation](./assets/convolutional_layer_backpropagation.jpg)
*Convolutional layer backpropagation*

## Dense layer
![Dense layer - Forwarding](./assets/dense_layer_forwarding.jpg)
*Dense layer forwarding - input (1x7), weights (7x4), biases (1x4)*

![Dense layer - Back-propagation](./assets/dense_layer_backpropagation.jpg)
*Dense layer backpropagation*

## Pooling layer
### Average pooling layer
![Average pooling layer - Forwarding](./assets/avg_pooling_layer_forwarding.jpg)
*Average pooling layer forwarding - input (4x4x3), kernel (2x2x3)*

![Average pooling layer - Back-propagation](./assets/avg_pooling_layer_backpropagation.jpg)
### Max pooling layer
![Max pooling layer - Forwarding](./assets/max_pooling_layer_forwarding.jpg)
*Max pooling layer forwarding - input (4x4x3), kernel (2x2x3)*

![Max pooling layer - Forwarding (example)](./assets/max_pooling_layer_forwarding_example.jpg)
![Average pooling layer - Back-propagation](./assets/max_pooling_layer_backpropagation.jpg)
### Softmax layer
![Softmax layer - Forwarding](./assets/softmax_layer_forwarding.jpg)
![Softmax layer - Back-propagation](./assets/softmax_layer_backpropagation.jpg)
