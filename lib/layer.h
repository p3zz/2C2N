#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "common.h"

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
}conv_layer_t;

typedef struct{
	int kernel_size;
	int stride;
	int padding;
	pooling_type type;
}pool_layer_t;

typedef struct{
	matrix2d_t weights;
	matrix2d_t biases;
}dense_layer_t;

typedef struct
{
	int neurons_num;
	neuron_t* neurons;
} layer_t;

layer_t create_layer(int num_neurons);
void destroy_layer(layer_t* layer);
void init_conv_layer(
	conv_layer_t* layer,
	int kernel_size,
	int kernel_depth,
	int kernels_n,
	int stride,
	int padding
);
void destroy_conv_layer(conv_layer_t* layer);
void process_conv_layer(const conv_layer_t* const layer, const matrix3d_t* const input, matrix3d_t* output);

void init_pool_layer(pool_layer_t* layer, int kernel_size, int padding, int stride, pooling_type type);
void process_pool_layer(const pool_layer_t* const layer, const matrix3d_t* const input, matrix3d_t* output);

void init_dense_layer(dense_layer_t* layer, int input_n, int output_n);
void process_dense_layer(const dense_layer_t* const layer, const matrix2d_t* const input, matrix2d_t* output);

#endif