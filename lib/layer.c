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
	int kernel_size,
	int kernel_depth,
	int stride,
	int padding)
{

	layer->kernel = (matrix3d_t*)malloc(sizeof(matrix3d_t));

	create_matrix3d(layer->kernel, kernel_size, kernel_size, kernel_depth);

	for(int i=0;i<layer->kernel->depth;i++){
		for(int j=0;j<layer->kernel->layers[i].rows_n;j++){
			for(int k=0;k<layer->kernel->layers[i].cols_n;k++){
            	layer->kernel->layers[i].values[j][k] = ((double)rand())/((double)RAND_MAX);				
			}
		}
	}
}

void destroy_layer(layer_t* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}