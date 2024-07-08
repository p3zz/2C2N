#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utils.h"
#include "network.h"

#define LAYERS_NUM 4
#define ITERATIONS_NUM 20000
#define LEARNING_RATE 0.1

static network_t network;
static int neurons_per_layer[LAYERS_NUM] = {3, 4, 5, 4};

// each training sample has an array of values, one for each neuron of the input layer
// inputs[TRAINING_EXAMPLES_NUM][INPUT_NEURONS_NUM]
// EX: [[1, 2], [4, 1], [5, 8]]
static float **input;
static int inputs_num;

// each training sample has an array of desired outputs, one for each neuron of the output layer
// desired_outputs[TRAINING_EXAMPLES_NUM][OUTPUT_NEURONS_NUM]
static float **desired_outputs;
static int outputs_num;

// number of training samples
static int num_training_ex;

int main(void)
{
    // TODO initialize inputs
    // TODO initialize output targets

    if(create_network(LAYERS_NUM, neurons_per_layer) == ERR)
    {
        printf("Error in creating network...\n");
        return ERR;
    }

    for(int it=0;it<ITERATIONS_NUM;it++)
    {
        for(int i=0;i<num_training_ex;i++)
        {
            if(train(&network, input[i], inputs_num, desired_outputs[i], outputs_num, LEARNING_RATE) == ERR){
                return ERR;
            }
        }
    }

    destroy_network(&network);

    return SUCCESS;
}
