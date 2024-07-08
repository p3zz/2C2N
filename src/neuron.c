#include "neuron.h"
#include <stdlib.h>

neuron_t create_neuron(int weights_num)
{
	neuron_t neuron;

	neuron.actv = 0.0;
	neuron.weights = (float*) malloc(weights_num * sizeof(float));
	neuron.bias = 0.0;
	neuron.z = 0.0;

	neuron.dactv = 0.0;
	neuron.dw = (float*) malloc(weights_num * sizeof(float));
	neuron.dbias = 0.0;
	neuron.dz = 0.0;

	return neuron;
}

// TODO:
// Add destructor