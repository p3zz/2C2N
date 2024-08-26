#include "network.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

// Create Neural Network Architecture
network_t create_network(
    int layers_num,
    const int* neurons_per_layer,
    const activation_function* f_per_layer,
    const activation_function* df_per_layer
){
    network_t network;

    network.layers_num = layers_num;
    network.layers = (layer_t_old*) malloc(layers_num * sizeof(layer_t_old));

    for(int i=0;i<network.layers_num;i++)
    {
        // initialize layer
        layer_t_old* current_layer = &network.layers[i];
        *current_layer = create_layer(neurons_per_layer[i]);
        current_layer->neurons_num = neurons_per_layer[i];

        // initialize each neuron of the layer
        for(int j=0;j<neurons_per_layer[i];j++){
            int weights_num = 0;
            if(i < network.layers_num - 1){
                weights_num = neurons_per_layer[i+1];
            }
            neuron_t* current_neuron = &(current_layer->neurons[j]);
            *current_neuron = create_neuron(weights_num, f_per_layer[i], df_per_layer[i]);
            // initialize each weight of the neuron
            for(int k=0;k<weights_num;k++){
                current_neuron->weights[k] = ((double)rand())/((double)RAND_MAX);
                current_neuron->dw[k] = 0.0;
            }
            // initialize the bias of the neuron
            current_neuron->bias = ((double)rand())/((double)RAND_MAX);
            current_neuron->dbias = 0.0;
        }
    }

    return network;
}

// Back Propogate Error
int back_propagation(const network_t* network, const float* desired_outputs, int outputs_num)
{
    if(outputs_num != network->layers[network->layers_num-1].neurons_num){
        return ERR;
    }

    for(int i = network->layers_num; i >= 0; i--){
        layer_t_old* curr_layer = &network->layers[i];
        for(int j=0;j<curr_layer->neurons_num;j++){
            neuron_t* curr_neuron = &(curr_layer->neurons[j]);
            if(i == network->layers_num-1){
                // dC / da_i(L) = 2*(a_i - target_i)
                curr_neuron->dactv = 2*(curr_neuron->actv - desired_outputs[j]);
            }
            else{
                curr_neuron->dactv = 0.f;
                layer_t_old* next_layer = &network->layers[i+1];
                for(int k=0;k<next_layer->neurons_num;k++){
                    neuron_t* next_neuron = &next_layer->neurons[k];
                    curr_neuron->dactv += (curr_neuron->weights[k] * next_neuron->dactv_f(next_neuron->z) * next_neuron->dactv);
                    if(i > 0){
                        curr_neuron->dw[k] = curr_neuron->actv * next_neuron->dactv_f(next_neuron->z) * next_neuron->dactv;
                    }
                }
            }
            curr_neuron->dbias = curr_neuron->dactv_f(curr_neuron->z) * curr_neuron->dactv;
        }
    }
    return SUCCESS;
}

void forward_propagation(const network_t* network)
{
    // for each layer of the network
    // start from the second layer so we skip the input one
    for(int i=1;i<network->layers_num;i++)
    {   
        // for each neuron of the current layer
        for(int j=0;j<network->layers[i].neurons_num;j++)
        {
            neuron_t* current_neuron = &(network->layers[i].neurons[j]);
            
            current_neuron->z = current_neuron->bias;

            // for each neuron of the previous layer
            layer_t_old* previous_layer = &network->layers[i-1];
            
            for(int k=0;k<previous_layer->neurons_num;k++)
            {
                neuron_t* previous_neuron = &(previous_layer->neurons[k]);
                // update the output of each neuron of the current layer
                current_neuron->z += (previous_neuron->actv * previous_neuron->weights[j]);
            }

            // then apply to the output a given activation function            
            current_neuron->actv = current_neuron->actv_f(current_neuron->z);

            // printf("Forward : Layer %d neuron %d z %f a %f\n", i, j, current_neuron->z, current_neuron->actv);
            
        }
    }
}

void gradient_descents(const network_t* network, float learning_rate)
{
    // for each layer (excluding the output layer)
    for(int i=0;i<network->layers_num;i++)
    {
        // for each neuron of the current layer
        for(int j=0;j<network->layers[i].neurons_num;j++)
        {
            neuron_t* neuron = &network->layers[i].neurons[j];
            // update each weight of the current neuron
            neuron->bias = gradient_descent(neuron->bias, learning_rate,  neuron->dbias);

            if(i < network->layers_num - 1){
                for(int k=0;k<network->layers[i+1].neurons_num;k++)
                {
                    neuron->weights[k] = gradient_descent(neuron->weights[k], learning_rate, neuron->dw[k]);
                    // printf("Gradient descent : Layer %d neuron %d b %f w[%d] %f\n", i, j, neuron->bias, k, neuron->weights[k]);
                }
            }
        }
    }   
}

int check_values(const network_t* network){
    for(int i=0;i<network->layers_num;i++){
        for(int j=0;j<network->layers[i].neurons_num;j++){
            neuron_t* neuron = &network->layers[i].neurons[j];
            if(isnan(neuron->actv) || isnan(neuron->bias) || isnan(neuron->z)){
                return ERR;
            }
            for(int k=0;k<neuron->weights_num;k++){
                if(isnan(neuron->weights[k])){
                    return ERR;
                }
            }
        }
    }
    return SUCCESS;
}

// Feed inputs to input layer_t_old
int feed_input(const network_t* network, const float* inputs, int inputs_num)
{
    if(network->layers[0].neurons_num != inputs_num){
        return ERR;
    }
    for(int i=0;i<inputs_num;i++)
    {
        network->layers[0].neurons[i].actv = inputs[i];
    }
    return SUCCESS;
}

void destroy_network(network_t* network){
	for(int i=0;i<network->layers_num;i++){
        destroy_layer(&network->layers[i]);
	}
    free(network->layers);
}

// Compute Total Cost
int compute_cost(const network_t* network, const float* output_targets, int output_targets_num, float* result)
{
    layer_t_old* output_layer = &(network->layers[network->layers_num-1]);

    if(output_targets_num != output_layer->neurons_num){
        return ERR;
    }
    float total_cost = 0;

    for(int i=0;i<output_layer->neurons_num;i++)
    {
        float tmpcost = output_targets[i] - output_layer->neurons[i].actv;
        float cost = (tmpcost * tmpcost) / 2.f;
        total_cost += cost;
    }

    *result = total_cost;

    return SUCCESS;
}

int train(const network_t* network, const float* const inputs, int inputs_num, const float* const output_targets, int output_targets_num, float learning_rate, float* cost)
{
    if(feed_input(network, inputs, inputs_num) == ERR){
        return ERR;
    }
    forward_propagation(network);
    if(compute_cost(network, output_targets, output_targets_num, cost) == ERR){
        return ERR;
    }
    if(back_propagation(network, output_targets, output_targets_num) == ERR){
        return ERR;
    }
    gradient_descents(network, learning_rate);
    if(check_values(network) == ERR){
        return ERR;
    }
}
