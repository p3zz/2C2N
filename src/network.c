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
        network.layers[i] = create_layer(neurons_per_layer[i]);
        network.layers[i].neurons_num = neurons_per_layer[i];
        printf("Created Layer: %d\n", i+1);
        printf("Number of Neurons in Layer %d: %d\n", i+1,network.layers[i].neurons_num);

        for(int j=0;j<neurons_per_layer[i];j++)
        {
            if(i < (network.layers_num-1)) 
            {
                network.layers[i].neurons[j] = create_neuron(neurons_per_layer[i+1]);
            }

            printf("Neuron %d in Layer %d created\n",j+1,i+1);  
        }
        printf("\n");
    }

    printf("\n");

    // // Initialize the weights
    // if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    // {
    //     printf("Error Initilizing weights...\n");
    //     return 1;
    // }

    return 0;
}