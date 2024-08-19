#include "layer.h"
#include <stdlib.h>
#include "utils.h"

layer_t create_layer(int neurons_num)
{
	layer_t lay;
	lay.neurons_num = neurons_num;
	lay.neurons = (neuron_t*) malloc(neurons_num * sizeof(neuron_t));
	return lay;
}

void init_conv_layer(
	conv_layer_t* layer,
	int kernel_size,
	int kernel_depth,
	int stride,
	int padding)
{

	layer->kernel = (matrix3d_t*)malloc(sizeof(matrix3d_t));

	create_matrix3d(layer->kernel, kernel_size, kernel_size, kernel_depth);

	layer->padding = padding;
	layer->stride = stride;
}

void destroy_layer(layer_t* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}