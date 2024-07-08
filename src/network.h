#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct{
    layer_t* layers;
    int layers_num;
} network_t;

int create_network(int layers_num, const int* neurons_per_layer);
void forward_propagation(network_t* network);
int back_propagation(network_t* network, float* desired_outputs, int outputs_num);
void update_weights(network_t* network, float learning_rate);
int feed_input(network_t* network, float* inputs, int inputs_num);

#endif