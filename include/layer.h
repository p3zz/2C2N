#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "stdbool.h"
#include "utils.h"

typedef enum{
	POOLING_TYPE_AVERAGE,
	POOLING_TYPE_MAX
}pooling_type;

typedef struct{
	matrix3d_t* input;
	// kernels is an array of 3d matrices, with length kernels_n
	matrix3d_t* kernels;
	int kernels_n;
	// biases is an array of 2d matrices, with length kernels_n
	matrix2d_t* biases;
	int stride;
	int padding;
	activation_type activation_type;
	matrix3d_t* output;
	matrix3d_t* output_activated;
	matrix3d_t* d_input;
}conv_layer_t;

typedef struct{
	matrix3d_t input;
	int kernel_size;
	int stride;
	int padding;
	pooling_type type;
	matrix3d_t output;
	matrix3d_t* indexes;
	matrix3d_t d_input;
}pool_layer_t;

typedef struct{
	matrix3d_t inputs;
	matrix2d_t weights;
	matrix2d_t biases;
	activation_type activation_type;
	matrix3d_t output;
	matrix3d_t output_activated;
	matrix3d_t d_inputs;
}dense_layer_t;

typedef struct{
	matrix3d_t input;
	matrix3d_t d_input;
	matrix3d_t output;
}softmax_layer_t;

typedef union{
	conv_layer_t conv_layer;
	pool_layer_t pool_layer;
	dense_layer_t dense_layer;
}layer_t;

void conv_layer_init(
	conv_layer_t* layer,
	int input_height,
	int input_width,
	int input_depth,
	int kernel_size,
	int kernels_n,
	int stride,
	int padding,
	activation_type activation_type
);

void conv_layer_feed(conv_layer_t* layer, matrix3d_t* input);
void conv_layer_backpropagation(conv_layer_t* layer, const matrix3d_t* const input, float learning_rate);
void conv_layer_destroy(conv_layer_t* layer);
void conv_layer_forwarding(conv_layer_t* layer);

void pool_layer_init(pool_layer_t* layer, int input_height, int input_width, int input_depth, int kernel_size, int padding, int stride, pooling_type type);
void pool_layer_feed(pool_layer_t* layer, const matrix3d_t* const input);
void pool_layer_forwarding(pool_layer_t* layer);
void pool_layer_backpropagation(pool_layer_t* layer, const matrix3d_t* const input);
void pool_layer_destroy(pool_layer_t* layer);

void dense_layer_init(dense_layer_t* layer, int input_n, int output_n, activation_type activation_type);
void dense_layer_feed(dense_layer_t* layer, const matrix3d_t* const input);
void dense_layer_forwarding(dense_layer_t* layer);
void dense_layer_backpropagation(dense_layer_t* layer, const matrix3d_t* const input, float learning_rate);
void dense_layer_destroy(dense_layer_t* layer);

void softmax_layer_init(softmax_layer_t* layer, int input_n);
void softmax_layer_feed(softmax_layer_t* layer, const matrix3d_t* const input);
void softmax_layer_forwarding(softmax_layer_t* layer);
void softmax_layer_backpropagation(softmax_layer_t* layer, const matrix3d_t* const input);
void softmax_layer_destroy(softmax_layer_t* layer);

#endif