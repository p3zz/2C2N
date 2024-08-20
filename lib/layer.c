#include "layer.h"
#include <stdlib.h>
#include "utils.h"
#include "common.h"

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
	int kernels_n,
	int stride,
	int padding)
{
	layer->kernels_n = kernels_n;
	layer->kernels = (matrix3d_t*)malloc(layer->kernels_n * sizeof(matrix3d_t));

	for(int i=0;i<layer->kernels_n;i++){
		create_matrix3d(&layer->kernels[i], kernel_size, kernel_size, kernel_depth);
	}

	layer->padding = padding;
	layer->stride = stride;
}

// void feed_forward(conv_layer_t* layer, matrix3d_t* input, matrix3d_t* output){
// 	create_matrix3d(output, )
// 	for(int i=0;i<layer->kernel->depth;i++){
// 		for(int j=0;j<input->depth;j++){

// 		}
// 	}
// }

void destroy_conv_layer(conv_layer_t* layer){
	for(int i=0;i<layer->kernels_n;i++){
		destroy_matrix3d(&layer->kernels[i]);
	}
	free(layer->kernels);
}

void destroy_layer(layer_t* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}