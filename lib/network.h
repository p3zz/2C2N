#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct{
    layer_t_old* layers;
    int layers_num;
} network_t;

network_t create_network(
    int layers_num,
    const int* neurons_per_layer,
    const activation_function* f_per_layer,
    const activation_function* df_per_layer
);
void forward_propagation(const network_t* network);
int back_propagation(const network_t* network, const float* desired_outputs, int outputs_num);
void gradient_descents(const network_t* network, float learning_rate);
int feed_input(const network_t* network, const float* inputs, int inputs_num);
void destroy_network(network_t* network);
int compute_cost(const network_t* network, const float* output_targets, int output_targets_num, float* result);
int train(const network_t* network, const float* const inputs, int inputs_num, const float* const output_targets, int output_targets_num, float learning_rate, float* cost);

#endif