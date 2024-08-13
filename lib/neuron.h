#ifndef NEURON_H
#define NEURON_H

#include "utils.h"


typedef struct
{
	// value of the output
	float z;
	activation_function actv_f;
	activation_function dactv_f;
	// value of the activated output (Filtered by the activation function)
	float actv;
	// each neuron is fully connected to the next layer
	// it means that each neuron needs to carry a weight for every neuron of the next layer 
	float *weights;
	int weights_num;
	// bias of the neuron
	float bias;

	// correction of the output
	float dz;
	// correction of the activated output
	float dactv;
	// correction of each weight
	float *dw;
	// correction of the bias
	float dbias;

	// TODO: Add function pointer for destructor

} neuron_t;

neuron_t create_neuron(int num_out_weights, const activation_function actv_f, const activation_function dactv_f);
void destroy_neuron(neuron_t* neuron);

#endif