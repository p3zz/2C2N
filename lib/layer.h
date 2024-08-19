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

#endif