#include "layer.h"
#include <stdlib.h>

layer_t create_layer(int neurons_num)
{
	layer_t lay;
	lay.neurons_num = neurons_num;
	lay.neurons = (neuron_t*) malloc(neurons_num * sizeof(neuron_t));
	return lay;
}

void destroy_layer(layer_t* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}