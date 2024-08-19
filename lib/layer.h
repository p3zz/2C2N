#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "common.h"

typedef struct{
	matrix3d_t* kernel;
	int stride;
	int padding;
}conv_layer_t;

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
	int stride,
	int padding
);

#endif