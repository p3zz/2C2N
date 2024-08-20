#include "layer.h"
#include <stdlib.h>
#include "utils.h"
#include "common.h"
#include "stdbool.h"

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

// TODO check if depth of each kernel is equal to the number of channels of the input
void feed_forward(const conv_layer_t* const layer, const matrix3d_t* const input, matrix3d_t* output){
	matrix2d_t result = {0};
	output->depth = layer->kernels_n;
	output->layers = (matrix2d_t*)malloc(output->depth * sizeof(matrix2d_t));
	for(int i=0;i<layer->kernels_n;i++){
		for(int j=0;j<layer->kernels[i].depth;j++){
			// compute the cross correlation between a channel of the input and its corresponding kernel
			cross_correlation(&input->layers[j], &layer->kernels[i].layers[j], &result, layer->padding, layer->stride);
			//we only know here how big the result will be
			// if we're computing the first cross_correlation (between the first channel of the input and the kernel),
			// we need to allocate the memory for the result
			if(j == 0){
				create_matrix2d(&output->layers[i], result.rows_n, result.cols_n);
			}
			// then we sum the resulting matrix to the output
			matrix2d_sum_inplace(&result, &output->layers[i]);
			// and we free the result matrix
			destroy_matrix2d(&result);
		}
	}
}

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