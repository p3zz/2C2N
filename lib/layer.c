#include "layer.h"
#include <stdlib.h>

layer_t create_layer(int neurons_num)
{
	layer_t lay;
	lay.neurons_num = neurons_num;
	lay.neurons = (neuron_t*) malloc(neurons_num * sizeof(neuron_t));
	return lay;
}

void init_conv_layer(
	conv_layer_t* layer,
	int out_channels_n,
	int kernel_size,
	int stride,
	int padding)
{
	layer->kernel->depth = out_channels_n;
	for(int i=0;i<layer->kernel->depth;i++){
		create_matrix(&layer->kernel->layers[i], kernel_size, kernel_size);
	}
	layer->out_channels_n = out_channels_n;
}

void destroy_layer(layer_t* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}