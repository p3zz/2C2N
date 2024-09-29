# 2C2N - A CNN development framework, written in C

## Description

2C2N is a C-based framework, inspired by PyTorch, for building and running Convolutional Neural Networks (CNNs). It features implementations of commonly used CNN layers such as convolutional, pooling, and dense (or fully connected) layers, along with its related forwarding and back-propagation procedures. The framework also includes well-known computation functions and data structures like matrix2d/3d for efficient matrix operations. The code has been developed in order to provide an easy, intuitive and modular interface with a low footprint memory usage.

## Tools

### Build
The build process is performed using [CMake](https://cmake.org/).
To build the whole project (every target), run:
```bash
./tools/build.sh
#or
mkdir build
cd build
cmake ..
cmake --build . --target all
```
To clear the build folder, run:
```bash
./tools/clear.sh
#or
rm -r build
```

### Test
The whole project is tested using the [Unity](https://github.com/ThrowTheSwitch/Unity.git) framework,
whose repository has to be cloned as submodule:
```bash
git clone --recurse-submodules git@github.com:p3zz/c_neural_network.git

#or

git clone git@github.com:p3zz/c_neural_network.git
git submodule init
git submodule update
```
To run a test, run:
```bash
./tools/test.sh
# or
cd build/test
ctest -V
```

Each library (common, matrix, layer, utils) has been tested using 3 different test suites:
- test_common
- test_layer
- test_matrix

The core part of the testing is the layer suite, where almost every part of the framework is linked together to perform
some "free-runs" of perceptrons (and / or) and an MNIST network.

### Code formatting
The code style and formatting is performed using [CLang Format](https://clang.llvm.org/docs/ClangFormat.html).

To format the code, run:
```bash
./tools/format.sh
# or
clang-format -i {include,src}/*
clang-format -i test/test_*
```

### Memory check
The memory check of the project has been performed using [Valgrind](https://valgrind.org/docs/manual/mc-manual.html).
The memory has been checked in every test suite.

To perform a heap-check, run:
```bash
./tools/heap-check.sh
# or
valgrind --leak-check=full build/test/test_layer
valgrind --leak-check=full build/test/test_common
valgrind --leak-check=full build/test/test_matrix
```

## Snippet
Here's a snippet of code that shows a basic usage of the framework, creating a network of 3 layers:
- convolutional
- average pooling 
- dense

and performing forwarding and back-propagation.

```c
void test_doc_example(void) {
  const float learning_rate = 0.015;
  float error = 0.f;
  /* input matrix to the network*/
  matrix3d_t input = {0};
  /* derivative matrix of the cost of the network*/
  matrix3d_t d_error = {0};
  /* auxiliary matrix used to reshape the d between the dense layer and the pool
  layer during forwarding */
  matrix3d_t aux = {0};
  /* matrix containing the output targets */
  matrix2d_t output_target = {0};
  float output_target_values[] = {0.f, 1.f};

  matrix2d_t output_channel = {0};
  matrix2d_t d_error_channel = {0};

  conv_layer_t layer0 = {0};
  pool_layer_t layer1 = {0};
  dense_layer_t layer2 = {0};

  /* initialize 3D input matrix of (7x7x2) and randomize its content */
  matrix3d_init(&input, 7, 7, 2);
  matrix3d_randomize(&input);

  matrix3d_init(&d_error, 1, 2, 1);
  matrix3d_init(&aux, 1, 2, 2);
  matrix2d_load(&output_target, 1, 2, output_target_values);

  /* initialize convolutional layer - input (7x7x2), 2 kernels (4x4x2), stride
   * 1, padding 0, activation relu */
  conv_layer_init(&layer0, 7, 7, 2, 4, 2, 1, 0, ACTIVATION_TYPE_RELU);
  /* initialize avg pooling layer - input (4x4x2), kernel (2x2x2), padding 0,
   * stride 2 */
  pool_layer_init(&layer1, 4, 4, 2, 2, 0, 2, POOLING_TYPE_AVERAGE);
  /* initialize dense layer - input (1x8), output (1x2), activation tanh*/
  dense_layer_init(&layer2, 8, 2, ACTIVATION_TYPE_TANH);

  /* retrieve a reference of each channel of interesting matrixes so
  we can compute the cost function */
  matrix3d_get_channel_as_mut_ref(layer2.output, &output_channel, 0);
  matrix3d_get_channel_as_mut_ref(&d_error, &d_error_channel, 0);

  /* forwarding */
  conv_layer_feed(&layer0, &input);
  conv_layer_forwarding(&layer0);
  pool_layer_feed(&layer1, layer0.output_activated);
  pool_layer_forwarding(&layer1);
  /* reshape the pool layer output to fit the dense layer input */
  matrix3d_reshape(layer1.output, layer2.input);
  /* the feed has been done by the reshape */
  dense_layer_forwarding(&layer2);

  /* compute the loss */
  error = mean_squared_error(&output_channel, &output_target);
  printf("Error: %.3f\n", error);
  /* compute the loss derivative */
  mean_squared_error_derivative(&output_channel, &output_target,
                                &d_error_channel);

  /* back-propagation */
  dense_layer_backpropagation(&layer2, &d_error, learning_rate);
  matrix3d_reshape(layer2.d_input, &aux);
  pool_layer_backpropagation(&layer1, &aux);
  conv_layer_backpropagation(&layer0, layer1.d_input, learning_rate);

  conv_layer_destroy(&layer0);
  pool_layer_destroy(&layer1);
  dense_layer_destroy(&layer2);

  matrix3d_destroy(&input);
  matrix3d_destroy(&d_error);
  matrix3d_destroy(&aux);
}
```

## Core

### Matrix
![Matrix2D](./assets/matrix2d.jpg)
![Matrix3D](./assets/matrix3d.jpg)

The framework is built around 2 main structures: *matrix2d_t* and *matrix3d_t*, defined as follows:
```c
typedef struct {
  int height;
  int width;
  float *values;
  bool loaded;
} matrix2d_t;

typedef struct {
  int height;
  int width;
  int depth;
  float *values;
  bool loaded;
} matrix3d_t;

```
A matrix can be created as follows:
```c
/* initialize a matrix, and allocate dynamically the *values pointer */
void matrix2d_init(matrix2d_t *m, int height, int width);
void matrix3d_init(matrix2d_t *m, int height, int width);

/* initialize a matrix, and set the *values pointer to a valid base_address */
void matrix2d_load(matrix2d_t *m, int height, int width,
                   float *const base_address);
void matrix3d_load(matrix2d_t *m, int height, int width,
                   int depth, float *const base_address);
```
and can be destroyed as follows:
```c
/* destroy a matrix (frees the *values pointer, has no effect if the
matrix has beeninitialized with matrix2d_load(...))
*/
void matrix2d_destroy(const matrix2d_t *m);
void matrix3d_destroy(const matrix3d_t *m);
```

The content of the matrix is stored inside a flattened 1D array, but the interface is built
to get the user be able to manipulate the matrix in a *math_like* fashion:

```c
/* get a non-mutable reference to a cell of the matrix */
const float *matrix2d_get_elem_as_ref(const matrix2d_t *const m, int row_idx,
                                      int col_idx);
const float *matrix3d_get_elem_as_ref(const matrix3d_t *const m, int row_idx,
                                      int col_idx, int z_idx);

/* get a mutable reference to a cell of the matrix */
float *matrix3d_get_elem_as_mut_ref(const matrix3d_t *const m, int row_idx,
                                    int col_idx, int z_idx);
float *matrix2d_get_elem_as_mut_ref(const matrix2d_t *const m, int row_idx,
                                    int col_idx);

/* get the value of a cell of the matrix */
float matrix2d_get_elem(const matrix2d_t *const m, int row_idx, int col_idx);
float matrix3d_get_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                        int z_idx);

/* set the value of a cell of the matrix */
void matrix2d_set_elem(const matrix2d_t *const m, int row_idx, int col_idx,
                       float value);
void matrix3d_set_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                       int z_idx, float value);

/* get a channel of 3D matrix as a 2D matrix */
void matrix3d_get_channel_as_mut_ref(const matrix3d_t *m, matrix2d_t *result,
                                   int z_idx);

```

Furthermore, the framework expose a bunch of useful functions for matrix operations:
```c
// copy the content of *values* of the input matrix inside the output matriux
void matrix3d_copy_content(const matrix3d_t *const input,
                           const matrix3d_t *output);
void matrix2d_copy_content(const matrix2d_t *const input,
                           const matrix2d_t *output);

// randomize the content of *values* with floats between 0 and 1
void matrix2d_randomize(matrix2d_t *input);
void matrix3d_randomize(matrix3d_t *input);

// flip a matrix upwise-down
void matrix2d_rotate180_inplace(const matrix2d_t *const input);

// perform an element-wise product between two 2D matrixes
void matrix2d_element_wise_product_inplace(const matrix2d_t *const m1,
                                           const matrix2d_t *const m2);

// perform an element-wise sum between two 2D matrixes
void matrix2d_sum_inplace(const matrix2d_t *const m1,
                          const matrix2d_t *const m2);

// reshape a 3D matrix inside another 3D matrix with given height, width and depth
void matrix3d_reshape(const matrix3d_t *const input, matrix3d_t *output);

```

### Layers
The framework provides a common interface to build, run and destroy the most used
type of layers of a CNN:

- Convolutional layer
- Pooling layer
- Dense (fully connected) layer

The shared opeations that can be performed on a generic layer are:
- init
- feed
- forwarding
- back-propagation
- destroy

From a memory point-of-view, every inner member of a layer that will be used to perform
feed, forwarding and back-propagation are allocated during the init phase of the layer.

#### Convolutional layer
The **convolutional layer** is implemented using the *conv_layer* struct.
```c
typedef struct {
  matrix3d_t *input;
  matrix3d_t *kernels;
  matrix2d_t *biases;
  matrix3d_t *output;
  matrix3d_t *output_activated;
  matrix3d_t *d_input;
  int kernels_n;
  int stride;
  int padding;
  activation_type activation_type;
  bool loaded;
} conv_layer_t;
```

A **convolutional layer** can be created as follows:

```c
/* initialze a convolutional layer, and allocate dinamically every pointer of the struct */
void conv_layer_init(conv_layer_t *layer, int input_height, int input_width,
                     int input_depth, int kernel_size, int kernels_n,
                     int stride, int padding, activation_type activation_type);

/* initialze a convolutional layer, and set every pointer of the struct to the corresponding
argument*/
void conv_layer_init_load(conv_layer_t *layer, matrix3d_t *kernels,
                          int kernels_n, matrix2d_t *biases, matrix3d_t *output,
                          matrix3d_t *output_activated, matrix3d_t *d_input,
                          int stride, int padding,
                          activation_type activation_type);
```

and destroyed as follows:
```c
/* destroy a layer (frees every dynamically allocated inner member, has no effect 
if the layer has been created with conv_layer_init_load(...))*/
void conv_layer_destroy(conv_layer_t *layer);
```

The shared operations has been developed as follows:
```c
/* feed the layer (copy the content of the *values pointer of the input to the corresponding inner member
of the layer */
void conv_layer_feed(conv_layer_t *layer, matrix3d_t *input);

/* feed the layer (set the *values pointer of the corresponding inner member
of the layer to the input */
void conv_layer_feed_load(conv_layer_t *layer, matrix3d_t *const input);

/* perform the forwarding stage. The result will be available inside the *output* and *output_activated*
inner members */
void conv_layer_forwarding(conv_layer_t *layer);

/* perform the back-propagation stage. Every weight/bias will be corrected during this stage, and 
the result will be available inside the *d_input* inner member */
void conv_layer_backpropagation(conv_layer_t *layer,
                                const matrix3d_t *const input,
                                float learning_rate);

```

##### Forwarding
The formula used to compute the output matrix is:
$$
Y_i = B_i + \sum_{j=1}^{n} X_j * K_{ij}\\
Yactv_{i} = actv(Y_i)
$$
where:
- Y_i is the i-th channel of the output
- B_i is the i-th bias
- X_j is the j-th channel of the input
- K_ij is the j-th channel of the i-th kernel

In order to compute the i-th channel of the output matrix, we perform the sum of the cross_correlation between each j-th channel of the i-th kernel and the j-th channel of the input, then add the bias to it.

![Convolutional layer - Forwarding](./assets/convolutional_layer_forwarding.jpg)
*Convolutional layer forwarding - input (5x4x4), 2 kernels (3x3x3), 2 biases (3x2), padding 0, stride 1*

##### Back-propagation
The formula used to correct the kernels/biases and to compute the derivative of the error w.r.t. to the input are:
$$
\frac{dE}{dK_{ij}} = X_j * \frac{dE}{dY_i} *_{wise} \frac{dactv}{dY_i} \\
\frac{dE}{dB_i} = \frac{dE}{dY_i} *_{wise} \frac{dactv}{dY_i} \\
\frac{dE}{dXj} = \sum_{i=1}^{n} (\frac{dE}{dY_i} *_{wise} \frac{dactv}{dY_i} *_{full} K_{ij}) \\
$$
Each derivative (except for the derivative of the error w.r.t. the input) are then used to correct
the kernels/biases using the gradient descent:
$$
K_{ij} = K_{ij} - (\frac{dE}{dK_{ij}} * \alpha) \\
B_{i} = B_{i} - (\frac{dE}{dB_{i}} * \alpha)
$$
where alpha is the learning rate

![Convolutional layer - Back-propagation](./assets/convolutional_layer_backpropagation.jpg)
*Convolutional layer backpropagation*

#### Dense layer
The **dense layer** is implemented using the *dense_layer* struct.
```c
typedef struct {
  matrix3d_t *input;
  matrix2d_t *weights;
  matrix2d_t *biases;
  matrix3d_t *output;
  matrix3d_t *output_activated;
  matrix3d_t *d_input;
  activation_type activation_type;
  bool loaded;
} dense_layer_t;
```

A **dense layer** can be created as follows:

```c
/* initialze a dense layer, and allocate dinamically every pointer of the struct */
void dense_layer_init(dense_layer_t *layer, int input_n, int output_n,
                      activation_type activation_type);

/* initialze a dense layer, and set every pointer of the struct to the corresponding
argument*/
void dense_layer_init_load(dense_layer_t *layer, matrix2d_t *weights,
                           matrix2d_t *biases, matrix3d_t *output,
                           matrix3d_t *output_activated, matrix3d_t *d_input,
                           activation_type activation_type);
```

and destroyed as follows:
```c
/* destroy a layer (frees every dynamically allocated inner member, has no effect 
if the layer has been created with conv_layer_init_load(...))*/
void dense_layer_destroy(dense_layer_t *layer);
```

The shared operations has been developed as follows:
```c
/* feed the layer (copy the content of the *values pointer of the input to the corresponding inner member
of the layer */
void dense_layer_feed(dense_layer_t *layer, matrix3d_t *input);

/* feed the layer (set the *values pointer of the corresponding inner member
of the layer to the input */
void dense_layer_feed_load(dense_layer_t *layer, matrix3d_t *const input);

/* perform the forwarding stage. The result will be available inside the *output* and *output_activated*
inner members */
void dense_layer_forwarding(dense_layer_t *layer);

/* perform the back-propagation stage. Every weight/bias will be corrected during this stage, and 
the result will be available inside the *d_input* inner member */
void dense_layer_backpropagation(dense_layer_t *layer,
                                const matrix3d_t *const input,
                                float learning_rate);

```

##### Forwarding
The formula used to compute the output matrix is:
$$
Y = B + X * W \\
Yactv = actv(Y)
$$
where:
- Y is the output
- B are the biases
- X is the input
- W are the weights

In order to compute the output matrix, you need to perform a matrix multiplication between the input matrix and the weights matrix, then add the bias to it.

![Dense layer - Forwarding](./assets/dense_layer_forwarding.jpg)
*Dense layer forwarding - input (1x7), weights (7x4), biases (1x4)*

##### Back-propagation
The formula used to correct weights/biases, as well as the derivative of the error w.r.t. the input is:
$$
\frac{dE}{dW_{ij}} = X_i * \frac{dE}{dY_j} * \frac{dactv}{dY_j} \\
\frac{dE}{dB_i} = \frac{dE}{dY_i} * \frac{dactv}{dY_i} \\
\frac{dE}{dY_i} = \sum_{j=0}^{n} (\frac{dE}{dW_{ji}} * \frac{dE}{dY_i} * \frac{dactv}{dY_i}) \\
$$

The correction of the weights/biases are performed the same way as the convolutional layer
![Dense layer - Back-propagation](./assets/dense_layer_backpropagation.jpg)
*Dense layer backpropagation*

#### Pooling layer
The **pooling layer** is implemented using the *pool_layer* struct.
```c
typedef struct {
  matrix3d_t *input;
  matrix3d_t *output;
  matrix3d_t *indexes;
  matrix3d_t *d_input;
  int kernel_size;
  int stride;
  int padding;
  pooling_type type;
  bool loaded;
} pool_layer_t;
```

A **pooling layer** can be created as follows:

```c
/* initialze a pool layer, and allocate dinamically every pointer of the struct */
void pool_layer_init(pool_layer_t *layer, int input_height, int input_width,
                     int input_depth, int kernel_size, int padding, int stride,
                     pooling_type type);

/* initialze a pool layer, and set every pointer of the struct to the corresponding
argument*/
void pool_layer_init_load(pool_layer_t *layer, matrix3d_t *output,
                          matrix3d_t *d_input, matrix3d_t *indexes,
                          int kernel_size, int stride, int padding,
                          pooling_type type);
```

and destroyed as follows:
```c
/* destroy a layer (frees every dynamically allocated inner member, has no effect 
if the layer has been created with conv_layer_init_load(...))*/
void pool_layer_destroy(pool_layer_t *layer);
```

The shared operations has been developed as follows:
```c
/* feed the layer (copy the content of the *values pointer of the input to the corresponding inner member
of the layer */
void pool_layer_feed(pool_layer_t *layer, matrix3d_t *input);

/* feed the layer (set the *values pointer of the corresponding inner member
of the layer to the input */
void pool_layer_feed_load(pool_layer_t *layer, matrix3d_t *const input);

/* perform the forwarding stage. The result will be available inside the *output* and *output_activated*
inner members */
void pool_layer_forwarding(pool_layer_t *layer);

/* perform the back-propagation stage. The result will be available inside the *d_input* inner member */
void pool_layer_backpropagation(pool_layer_t *layer, const matrix3d_t *const input);
```

#### Average pooling layer
##### Forwarding
The formula used to compute the output matrix is:
$$
Y_{pqc} = \frac{1}{k^2} \sum_{i=0}^{k-1} \sum_{j=0}^{k-1}X_{shp+i, swq+j, c}
$$
where:
- Y_pqc is the output value at index pqc
- k is the size of the edge of the kernel (the kernel is squared)
- p and q are the indices for the output feature map
- c is the channel index
- *shp* and *swq* represent the top-left corner of the pooling window applied to the input feature map.

![Average pooling layer - Forwarding](./assets/avg_pooling_layer_forwarding.jpg)
*Average pooling layer forwarding - input (4x4x3), kernel (2x2x3), padding 0, stride 2*

##### Back-propagation
The back-propagation is pretty easy in this case. We need to propagate the derivative of the error w.r.t. the output only in the portion of the input that has been involved in the computation of a specific value of the output matrix.
![Average pooling layer - Back-propagation](./assets/avg_pooling_layer_backpropagation.jpg)

#### Max pooling layer
##### Forwarding
The formula used to compute the output matrix is:
$$
Y_{pqc} = max_{0 \leq i \leq k_h, 0 \leq j \leq k_w} X_{shp + i, swq + j, c}
$$
where the nomenclature is the same as the average pooling layer.
![Max pooling layer - Forwarding](./assets/max_pooling_layer_forwarding.jpg)
*Max pooling layer forwarding - input (4x4x3), kernel (2x2x3), padding 0, stride 2*

During the forwarding stage, we need to remember the position of a specific value of the output matrix w.r.t. the input matrix, so during the back-propagation we already know which values of the input matrix have affected the following layers. To do this, for each channel of the output matrix, we use a 3D matrix in which, on the 1st channel, we keep track of the row index of the element that is referring to, while in the 2nd channel we keep track of the column index of the same element.
![Max pooling layer - Forwarding (example)](./assets/max_pooling_layer_forwarding_example.jpg)
*Max pooling layer forwarding example - input (4x4x1), kernel (2x2x1), padding 0, stride 2*

##### Back-propagation
During the back-propagation, we compute a derivative matrix in which we propagate backward the derivative of the error w.r.t. the output, only if the position of the input element is found inside the indexes matrices.
![Max pooling layer - Back-propagation](./assets/max_pooling_layer_backpropagation.jpg)

#### Softmax layer
The **softmax layer** is implemented using the *softmax_layer* struct.
```c
typedef struct {
  matrix3d_t *input;
  matrix3d_t *d_input;
  matrix3d_t *output;
  bool loaded;
} softmax_layer_t;
```

A **softmax layer** can be created as follows:

```c
/* initialze a pool layer, and allocate dinamically every pointer of the struct */
void softmax_layer_init(softmax_layer_t *layer, int input_n);

/* initialze a pool layer, and set every pointer of the struct to the corresponding
argument*/
void softmax_layer_init_load(softmax_layer_t *layer, matrix3d_t *output,
                             matrix3d_t *d_input);
```

and destroyed as follows:
```c
/* destroy a layer (frees every dynamically allocated inner member, has no effect 
if the layer has been created with conv_layer_init_load(...))*/
void softmax_layer_destroy(softmax_layer_t *layer);
```

The shared operations has been developed as follows:
```c
/* feed the layer (copy the content of the *values pointer of the input to the corresponding inner member
of the layer */
void softmax_layer_feed(softmax_layer_t *layer, matrix3d_t *input);

/* feed the layer (set the *values pointer of the corresponding inner member
of the layer to the input */
void softmax_layer_feed_load(softmax_layer_t *layer, matrix3d_t *const input);

/* perform the forwarding stage. The result will be available inside the *output* and *output_activated*
inner members */
void softmax_layer_forwarding(softmax_layer_t *layer);

/* perform the back-propagation stage. The result will be available inside the *d_input* inner member */
void softmax_layer_backpropagation(softmax_layer_t *layer, const matrix3d_t *const input);
```


##### Forwarding
The formula to compute the output matrix is:
$$
\sigma(Y_{ij}) = \frac{e^{Y_{ij}}}{\sum_{j=0}^{n-1} \sum_{k=0}^{m-1} e^{Y_{jk}}}
$$
where Y_ij is the value of the 2D output matrix.

![Softmax layer - Forwarding](./assets/softmax_layer_forwarding.jpg)
##### Back-propagation

![Softmax layer - Back-propagation](./assets/softmax_layer_backpropagation.jpg)
