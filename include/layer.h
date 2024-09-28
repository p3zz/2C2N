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
 * @param indexes: pointer to multiple 3D matrices used to keep track of the indeces of the maximum values
 * found during the feed-forward stage of a max pooling layer. For each slice of the output matrix, a 3D matrix
 * with depth = 2 (slice 0 to save row indeces, slice 1 to save column indeces) is used to keep track of the indeces
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
 * @param input: pointer to the input of the layer. This is used during the feed-forward stage
 * @param weights: pointer to the weights of the layer. Weights are stored inside a 2D matrix, 
 * where the width is equal to the height of the output, and the height is equal to the width of the input
 * @param biases: pointer to the biases of the layer. Biases are stored inside a 2D matrix,
 * where the width is equal to the height of the output, and the height is equal to 1
 * @param output: pointer to the output of the layer. This is computed after the feed-forward stage
 * @param output_activated: pointer to the activated output of the layer. This is computed as the
 * result of element-wise operation applied to the output of the layer
 * @param d_input: pointer to the derivative of the input w.r.t. the output. This is computed after the
 * back-proagation of the layer
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

void conv_layer_init(conv_layer_t *layer, int input_height, int input_width,
                     int input_depth, int kernel_size, int kernels_n,
                     int stride, int padding, activation_type activation_type);

void conv_layer_feed(conv_layer_t *layer, matrix3d_t *input);
void conv_layer_backpropagation(conv_layer_t *layer,
                                const matrix3d_t *const input,
                                float learning_rate);
void conv_layer_destroy(conv_layer_t *layer);
void conv_layer_forwarding(conv_layer_t *layer);

void pool_layer_init(pool_layer_t *layer, int input_height, int input_width,
                     int input_depth, int kernel_size, int padding, int stride,
                     pooling_type type);
void pool_layer_feed(pool_layer_t *layer, const matrix3d_t *const input);
void pool_layer_forwarding(pool_layer_t *layer);
void pool_layer_backpropagation(pool_layer_t *layer,
                                const matrix3d_t *const input);
void pool_layer_destroy(pool_layer_t *layer);

void dense_layer_init(dense_layer_t *layer, int input_n, int output_n,
                      activation_type activation_type);
void dense_layer_feed(dense_layer_t *layer, const matrix3d_t *const input);
void dense_layer_forwarding(dense_layer_t *layer);
void dense_layer_backpropagation(dense_layer_t *layer,
                                 const matrix3d_t *const input,
                                 float learning_rate);
void dense_layer_destroy(dense_layer_t *layer);

void softmax_layer_init(softmax_layer_t *layer, int input_n);
void softmax_layer_feed(softmax_layer_t *layer, const matrix3d_t *const input);
void softmax_layer_forwarding(softmax_layer_t *layer);
void softmax_layer_backpropagation(softmax_layer_t *layer,
                                   const matrix3d_t *const input);
void softmax_layer_destroy(softmax_layer_t *layer);

#endif