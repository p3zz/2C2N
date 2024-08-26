#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "common.h"
#include "stdbool.h"

typedef enum{
	POOLING_TYPE_AVERAGE,
	POOLING_TYPE_MAX
}pooling_type;

typedef struct{
	matrix3d_t* kernels;
	int kernels_n;	
	matrix2d_t* biases;
	int stride;
	int padding;
	activation_type activation_type;
	matrix3d_t output;
	matrix3d_t output_activated;
}conv_layer_t;

typedef struct{
	int kernel_size;
	int stride;
	int padding;
	pooling_type type;
	matrix3d_t output;
}pool_layer_t;

typedef struct{
	matrix2d_t inputs;
	matrix2d_t weights;
	matrix2d_t biases;
	activation_type activation_type;
	matrix2d_t output;
	matrix2d_t output_activated;
	matrix2d_t d_inputs;
}dense_layer_t;

typedef union{
	conv_layer_t conv_layer;
	pool_layer_t pool_layer;
	dense_layer_t dense_layer;
}layer_t;

typedef struct
{
	int neurons_num;
	neuron_t* neurons;
} layer_t_old;

layer_t_old create_layer(int num_neurons);
void destroy_layer(layer_t_old* layer);

void init_conv_layer(
	conv_layer_t* layer,
	int kernel_size,
	int kernel_depth,
	int kernels_n,
	int stride,
	int padding,
	activation_type activation_type
);
void destroy_conv_layer(conv_layer_t* layer);
void process_conv_layer(conv_layer_t* layer, const matrix3d_t* const input);

void init_pool_layer(pool_layer_t* layer, int kernel_size, int padding, int stride, pooling_type type);
void process_pool_layer(pool_layer_t* layer, const matrix3d_t* const input);
void destroy_pool_layer(pool_layer_t* layer);

void init_dense_layer(dense_layer_t* layer, int input_n, int output_n, activation_type activation_type);
void feed_dense_layer(dense_layer_t* layer, const matrix2d_t* const input);
void process_dense_layer(dense_layer_t* layer);
void backpropagation_dense_layer(dense_layer_t* layer, const matrix2d_t* const input, float learning_rate);
void destroy_dense_layer(dense_layer_t* layer);

#endif