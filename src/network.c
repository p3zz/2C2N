#include "network.h"
#include <stdlib.h>
#include <stdio.h>

// Create Neural Network Architecture
int create_network(int layers_num, const int* neurons_per_layer)
{
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
            neuron_t* current_neuron = &(current_layer->neurons[j]);
            *current_neuron = create_neuron(network.layers[i+1].neurons_num);
            // initialize each weight of the neuron
            for(int k=0;k<network.layers[i+1].neurons_num;k++){
                current_neuron->weights[k] = ((double)rand())/((double)RAND_MAX);
                current_neuron->dw[k] = 0.0;
            }
            // initialize the bias of the neuron
            current_neuron->bias = ((double)rand())/((double)RAND_MAX);
            current_neuron->dbias = 0.0;
        }
    }

    return 0;
}