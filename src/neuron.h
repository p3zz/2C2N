#ifndef NEURON_H
#define NEURON_H

typedef struct
{
	float actv;
	float *weights;
	float bias;
	float z;

	float dactv;
	float *dw;
	float dbias;
	float dz;

	// TODO: Add function pointer for destructor

} neuron_t;

neuron_t create_neuron(int num_out_weights);

#endif