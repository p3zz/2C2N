/**
 * @brief This library provides a basic implementation of the most common layers used in Convolutional Neural Networks (CNNs),
 * such as convolutional layers, dense layers and pooling layers.
 * For each type of layer, the library provides a common interface to perform the main operations used in CNN, such as:
 * - Feeding
 * - Forwarding
 * - Back-propagation
 */

#ifndef __LAYER_H__
#define __LAYER_H__

#include "common.h"

/**
 * @struct conv_layer_t
 * @brief Implementation of a convolutional layer
 * @param input: pointer to the input of the layer. This is used during the feed-forward stage
 * @param kernels: pointer to the kernels of the layer. The layer supports multiple 3D kernels.
 * @param biases: pointer to the biases of the layer. The layer supports multiple 3D biases,
 * @param output: pointer to the output of the layer. This is computed after the feed-forward stage
 * @param output_activated: pointer to the activated output of the layer. This is computed as the
 * result of element-wise operation applied to the output of the layer
 * @param d_input: pointer to the derivative of the input w.r.t. the output. This is computed after the
 * back-proagation of the layer
 * @param kernels_n: n. of kernels
 * @param stride: stride that will be applied to the convolution of each layer
 * @param padding: padding that will be applied to the convolution of each layer
 * @param activation_type: the type of activation used by the layer to compute the activated output.
 * @param loaded: flag used to keep track of the origin of the pointers.
 */
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

/**
 * @struct pool_layer_t
 * @brief Implementation of a pooling layer
 * @param input: pointer to the input of the layer. This is used during the feed-forward stage
 * @param output: pointer to the output of the layer. This is computed after the feed-forward stage
 * @param indexes: pointer to multiple 3D matrices used to keep track of the indexes of the maximum values
 * found during the feed-forward stage of a max pooling layer. For each slice of the output matrix, a 3D matrix
 * with depth = 2 (slice 0 to save row indexes, slice 1 to save column indexes) is used to keep track of the indexes
 * of the values contained in the output matrix w.r.t. the input matrix.
 * @param d_input: pointer to the derivative of the input w.r.t. the output. This is computed after the
 * @param kernel_size: the size of the kernel side (the pooling is performed using a squared kernel of size size*size)
 * @param stride: stride that will be applied to the pooling of each layer
 * @param padding: padding that will be applied to the pooling of each layer
 * @param type: the type of pooling used by the layer to compute the output.
 * @param loaded: flag used to keep track of the origin of the pointers.
 */
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

/**
 * @struct dense_layer_t
 * @brief Implementation of a dense layer
 * @param input: pointer to the input of the layer. This is used during the feed-forward stage.
 * The input matrix has height = 1, depth = 1, and width equals to the n. of inputs.
 * It's like a linear 1D array, but for compatibility with other layers it's implemented as a 3D matrix.
 * @param weights: pointer to the weights of the layer. Weights are stored inside a 2D matrix, 
 * where the width is equal to the height of the output, and the height is equal to the width of the input
 * @param biases: pointer to the biases of the layer. Biases are stored inside a 2D matrix,
 * where the width is equal to the height of the output, and the height is equal to 1
 * @param output: pointer to the output of the layer. This is computed after the feed-forward stage
 * The output matrix has height = 1, depth = 1, and width equals to the n. of outputs.
 * It's a 1D array, but for compatibility with other layers it's implemented as a 3D matrix.
 * @param output_activated: pointer to the activated output of the layer. This is computed as the
 * result of element-wise activation applied to the output of the layer.
 * The output_activated matrix has height = 1, depth = 1, and width equals to the n. of outputs.
 * It's a 1D array, but for compatibility with other layers it's implemented as a 3D matrix.
 * @param d_input: pointer to the derivative of the input w.r.t. the output. This is computed after the
 * back-proagation of the layer.
 * The d_input matrix has height = 1, depth = 1, and width equals to the n. of inputs.
 * It's a 1D array, but for compatibility with other layers it's implemented as a 3D matrix.
 * @param activation_type: the type of activation used by the layer to compute the activated output.
 * @param loaded: flag used to keep track of the origin of the pointers.
 */
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

/**
 * @struct softmax_layer_t
 * @brief Implementation of a softmax layer
 * @param input: pointer to the input of the layer. This is used during the feed-forward stage
 * @param output: pointer to the output of the layer. This is computed after the feed-forward stage
 * @param d_input: pointer to the derivative of the input w.r.t. the output. This is computed after the
 * back-proagation of the layer
 * @param loaded: flag used to keep track of the origin of the pointers.
 */
typedef struct {
  matrix3d_t *input;
  matrix3d_t *d_input;
  matrix3d_t *output;
  bool loaded;
} softmax_layer_t;


/**
 * @brief Initialize a convolutional layer. Each member of the layer is dinamically allocated.
 * @param layer: pointer to the target layer
 * @param input_height: height of the input
 * @param input_width: width of the input
 * @param input_depth: depth of the input
 * @param kernel_size: length of the kernel side
 * @param kernels_n: n. of kernels
 * @param stride: stride
 * @param padding: padding
 * @param activation_type: the type of activation used to activate the output
 */
void conv_layer_init(conv_layer_t *layer, int input_height, int input_width,
                     int input_depth, int kernel_size, int kernels_n,
                     int stride, int padding, activation_type activation_type);


/**
 * @brief Initialize a convolutional layer.
 * @param layer: pointer to the target layer
 * @param kernels: pointer to the kernels
 * @param kernels_n: n. of kernels
 * @param biases: pointer to the biases
 * @param output: pointer to the output
 * @param output_activated: pointer to the output activated
 * @param d_input: pointer to the d_input
 * @param stride: stride
 * @param padding: padding
 * @param activation_type: the type of activation used to activate the output
 */
void conv_layer_load_params(conv_layer_t *layer, matrix3d_t *kernels,
                            int kernels_n, matrix2d_t *biases,
                            matrix3d_t *output, matrix3d_t *output_activated,
                            matrix3d_t *d_input, int stride, int padding, activation_type activation_type);

/**
 * @brief Feed a convolutional layer, and copy the content of a 3D matrix inside the "input"
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void conv_layer_feed(conv_layer_t *layer, matrix3d_t *input);

/**
 * @brief Feed a convolutional layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void conv_layer_feed_load(conv_layer_t *layer, matrix3d_t *const input);

/**
 * @brief Perform the back-propagation stage of a convolutional layer.
 * @param layer: pointer to the target layer
 * @param input: derivative of the input w.r.t. the output coming from the next layer
 * @param learning_rate: the learning rate that corrects weights and biases
 */
void conv_layer_backpropagation(conv_layer_t *layer,
                                const matrix3d_t *const input,
                                float learning_rate);
/**
 * @brief Destroy a convolutional layer. Each member of the layer will be freed.
 * @param layer: pointer to the target layer
 */
void conv_layer_destroy(conv_layer_t *layer);

/**
 * @brief Perform the forwarding stage of a convolutional layer.
 * @param layer: pointer to the target layer
 */
void conv_layer_forwarding(conv_layer_t *layer);

/**
 * @brief Initialize a convolutional layer. Each member of the layer is dinamically allocated.
 * @param layer: pointer to the target layer
 * @param input_height: height of the input
 * @param input_width: width of the input
 * @param input_depth: depth of the input
 * @param kernel_size: length of the kernel side
 * @param stride: stride used to perform the pooling operation
 * @param padding: padding used to perform the pooling operation
 * @param type: the type of pooling used to compute the output
 */
void pool_layer_init(pool_layer_t *layer, int input_height, int input_width,
                     int input_depth, int kernel_size, int padding, int stride,
                     pooling_type type);

/**
 * @brief Initialize a pooling layer.
 * @param layer: pointer to the target layer
 * @param output: pointer to the output
 * @param d_input: pointer to the d_input
 * @param indexes: pointer to indexes
 * @param kernel_size: length of the kernel side
 * @param stride: stride used to perform the pooling operation
 * @param padding: padding used to perform the pooling operation
 * @param pooling_type: the type of pooling used to compute the output
 */
void pool_layer_load_params(pool_layer_t *layer, matrix3d_t *output, matrix3d_t *d_input, matrix3d_t *indexes,
int kernel_size, int stride, int padding, pooling_type type);

/**
 * @brief Feed a pooling layer, and copy the content of a 3D matrix inside the "input"
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void pool_layer_feed(pool_layer_t *layer, const matrix3d_t *const input);

/**
 * @brief Feed a pooling layer
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void pool_layer_feed_load(pool_layer_t *layer, matrix3d_t *const input);

/**
 * @brief Perform the forwarding stage of a pooling layer.
 * @param layer: pointer to the target layer
 */
void pool_layer_forwarding(pool_layer_t *layer);

/**
 * @brief Perform the back-propagation stage of a pooling layer.
 * @param layer: pointer to the target layer
 * @param input: derivative of the input w.r.t. the output coming from the next layer
 * @param learning_rate: the learning rate that corrects weights and biases
 */
void pool_layer_backpropagation(pool_layer_t *layer,
                                const matrix3d_t *const input);

/**
 * @brief Destroy a pooling layer. Each member of the layer will be freed.
 * @param layer: pointer to the target layer
 */
void pool_layer_destroy(pool_layer_t *layer);

/**
 * @brief Initialize a dense layer. Each member of the layer is dinamically allocated.
 * @param layer: pointer to the target layer
 * @param input_n: length of the input
 * @param output_n: length of the output
 * @param activation_type: the type of activation function used to compute the output
 */
void dense_layer_init(dense_layer_t *layer, int input_n, int output_n,
                      activation_type activation_type);

/**
 * @brief Initialize a convolutional layer.
 * @param layer: pointer to the target layer
 * @param weights: pointer to the biases
 * @param biases: pointer to the biases
 * @param output: pointer to the output
 * @param output_activated: pointer to the output activated
 * @param d_input: pointer to the d_input
 * @param activation_type: the type of activation used to activate the output
 */
void dense_layer_load_params(dense_layer_t *layer, matrix2d_t *weights,
                             matrix2d_t *biases, matrix3d_t *output,
                             matrix3d_t *output_activated,
                             matrix3d_t *d_input, activation_type activation_type);

/**
 * @brief Feed a dense layer, and copy the content of a 3D matrix inside the "input"
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void dense_layer_feed(dense_layer_t *layer, const matrix3d_t *const input);

/**
 * @brief Feed a dense layer
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void dense_layer_feed_load(dense_layer_t *layer, matrix3d_t *const input);

/**
 * @brief Perform the forwarding stage of a dense layer.
 * @param layer: pointer to the target layer
 */
void dense_layer_forwarding(dense_layer_t *layer);

/**
 * @brief Perform the back-propagation stage of a dense layer.
 * @param layer: pointer to the target layer
 * @param input: derivative of the input w.r.t. the output coming from the next layer
 * @param learning_rate: the learning rate that corrects weights and biases
 */
void dense_layer_backpropagation(dense_layer_t *layer,
                                 const matrix3d_t *const input,
                                 float learning_rate);

/**
 * @brief Destroy a dense layer. Each member of the layer will be freed.
 * @param layer: pointer to the target layer
 */
void dense_layer_destroy(dense_layer_t *layer);

/**
 * @brief Initialize a softmax layer. Each member of the layer is dinamically allocated.
 * @param layer: pointer to the target layer
 * @param input_n: length of the input
 */
void softmax_layer_init(softmax_layer_t *layer, int input_n);

/**
 * @brief Initialize a softmax layer.
 * @param layer: pointer to the target layer
 * @param output: pointer to the output
 * @param d_input: pointer to the d_input
 */
void softmax_layer_load_params(softmax_layer_t *layer, matrix3d_t *output,
                            matrix3d_t *d_input);

/**
 * @brief Feed a softmax layer, and copy the content of a 3D matrix inside the "input"
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void softmax_layer_feed(softmax_layer_t *layer, const matrix3d_t *const input);

/**
 * @brief Feed a softmax layer
 * member of the layer
 * @param layer: pointer to the target layer
 * @param input: pointer to the input to copy from 
 */
void softmax_layer_feed_load(softmax_layer_t *layer, matrix3d_t *const input);

/**
 * @brief Perform the forwarding stage of a softmax layer.
 * @param layer: pointer to the target layer
 */
void softmax_layer_forwarding(softmax_layer_t *layer);

/**
 * @brief Perform the back-propagation stage of a softmax layer.
 * @param layer: pointer to the target layer
 * @param input: derivative of the input w.r.t. the output coming from the next layer
 */
void softmax_layer_backpropagation(softmax_layer_t *layer,
                                   const matrix3d_t *const input);

/**
 * @brief Destroy a softmax layer. Each member of the layer will be freed.
 * @param layer: pointer to the target layer
 */
void softmax_layer_destroy(softmax_layer_t *layer);

#endif