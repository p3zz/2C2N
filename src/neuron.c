#include "neuron.h"
#include <stdlib.h>

neuron_t create_neuron(int num_out_weights)
{
	neuron_t neuron;

	neuron.actv = 0.0;
	neuron.weights = (float*) malloc(num_out_weights * sizeof(float));
	neuron.bias = 0.0;
	neuron.z = 0.0;

	neuron.dactv = 0.0;
	neuron.dw = (float*) malloc(num_out_weights * sizeof(float));
	neuron.dbias = 0.0;
	neuron.dz = 0.0;

	return neuron;
}

// TODO:
// Add destructor