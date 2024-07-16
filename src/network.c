#include "network.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "time.h"

// Create Neural Network Architecture
network_t create_network(int layers_num, const int* neurons_per_layer)
{
    srand(time(NULL));
    network_t network;

    network.layers_num = layers_num;
    network.layers = (layer_t*) malloc(layers_num * sizeof(layer_t));

    for(int i=0;i<network.layers_num;i++)
    {
        // initialize layer
        layer_t* current_layer = &network.layers[i];
        *current_layer = create_layer(neurons_per_layer[i]);
        current_layer->neurons_num = neurons_per_layer[i];

        // initialize each neuron of the layer
        for(int j=0;j<neurons_per_layer[i];j++){
            int weights_num = 0;
            if(i < network.layers_num - 1){
                weights_num = neurons_per_layer[i+1];
            }
            neuron_t* current_neuron = &(current_layer->neurons[j]);
            *current_neuron = create_neuron(weights_num);
            // initialize each weight of the neuron
            for(int k=0;k<weights_num;k++){
                current_neuron->weights[k] = ((double)rand())/((double)RAND_MAX);
                current_neuron->dw[k] = 0.0;
            }
            // initialize the bias of the neuron
            current_neuron->bias = 0.0;
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
    
    // for each layer iterate backwards from the output layer
    for(int i=network->layers_num-1;i>0;i--)
    {
        // for each neuron of the current layer
        for(int j=0;j<network->layers[i].neurons_num;j++)
        {
            neuron_t* current_neuron = &(network->layers[i].neurons[j]);
            // compute the partial derivative w.r.t. output
            if(i == network->layers_num-1){
                current_neuron->dz = (current_neuron->actv - desired_outputs[j]) * sigmoid_derivative(current_neuron->actv);
            }
            else{
                current_neuron->dz = relu_derivative(current_neuron->dactv);
            }

            // for each neuron of the previous layer
            for(int k=0;k<network->layers[i-1].neurons_num;k++)
            {
                neuron_t* previous_neuron = &(network->layers[i-1].neurons[k]);
                // compute the derivative w.r.t. weight
                previous_neuron->dw[j] = current_neuron->dz * previous_neuron->actv;
                
                // if the current layer is not the input layer
                if(i>1)
                {
                    // compute the derivative w.r.t. output
                    previous_neuron->dactv = current_neuron->dz * previous_neuron->weights[j];
                }
            }
            // compute the derivative w.r.t. bias
            current_neuron->dbias = current_neuron->dz;
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
            layer_t* previous_layer = &network->layers[i-1];
            
            for(int k=0;k<previous_layer->neurons_num;k++)
            {
                neuron_t* previous_neuron = &(previous_layer->neurons[k]);
                // update the output of each neuron of the current layer
                current_neuron->z += (previous_neuron->actv * previous_neuron->weights[j]);
            }

            // then apply to the output a given activation function based on the type of current layer
            // the last layer is the output layer, the previous ones are hidden layers because we are starting from the second layer (layer[1])
            // the first layer is the input layer
            bool is_hidden = i < (network->layers_num - 1);
            // Relu Activation Function for Hidden Layers
            if(is_hidden)
            {
                current_neuron->actv = relu(current_neuron->z);
            }
            
            // Sigmoid Activation function for Output Layer
            else
            {
                current_neuron->actv = sigmoid(current_neuron->z);
            }
            
        }
    }
}

void update_weights(const network_t* network, float learning_rate)
{
    // for each layer (excluding the output layer)
    for(int i=0;i<network->layers_num-1;i++)
    {
        // for each neuron of the current layer
        for(int j=0;j<network->layers[i].neurons_num;j++)
        {
            neuron_t* neuron = &network->layers[i].neurons[j];
            // update each weight of the current neuron
            for(int k=0;k<network->layers[i+1].neurons_num;k++)
            {
                neuron->weights[k] = update_weight(neuron->weights[k], learning_rate, neuron->dw[k]);
            }
            
            // Update Bias of the current neuron
            neuron->bias = update_bias(neuron->bias, learning_rate,  neuron->dbias);
        }
    }   
}

// Feed inputs to input layer_t
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
    layer_t* output_layer = &(network->layers[network->layers_num-1]);

    if(output_targets_num != output_layer->neurons_num){
        return ERR;
    }
    float total_cost = 0;

    for(int i=0;i<output_layer->neurons_num;i++)
    {
        float tmpcost = output_targets[i] - output_layer->neurons[i].actv;
        float cost = (tmpcost * tmpcost) / 2;
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
    update_weights(network, learning_rate);
}
