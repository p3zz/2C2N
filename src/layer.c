#include "layer.h"
#include <stdlib.h>

layer_t create_layer(int number_of_neurons)
{
	layer_t lay;
	lay.neurons_num = number_of_neurons;
	lay.neurons = (neuron_t*) malloc(number_of_neurons * sizeof(neuron_t));
	return lay;
}

void destroy_layer(layer_t* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
}