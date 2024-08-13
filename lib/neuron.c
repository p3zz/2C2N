#include "neuron.h"
#include <stdlib.h>
#include "utils.h"

neuron_t create_neuron(int weights_num, const activation_function f, const activation_function df)
{
	neuron_t neuron;

	neuron.actv = 0.0;
	neuron.actv_f = f;
	neuron.weights_num = weights_num;
	neuron.weights = (float*) malloc(neuron.weights_num * sizeof(float));
	neuron.bias = 0.0;
	neuron.z = 0.0;

	neuron.dactv = 0.0;
	neuron.dactv_f = df;
	neuron.dw = (float*) malloc(neuron.weights_num * sizeof(float));
	neuron.dbias = 0.0;
	neuron.dz = 0.0;

	return neuron;
}

void destroy_neuron(neuron_t* neuron){
	free(neuron->weights);
	free(neuron->dw);
}