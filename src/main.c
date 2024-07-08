#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utils.h"
#include "network.h"

#define SUCCESS 0
#define ERR 1
#define LAYERS_NUM 4
#define ITERATIONS_NUM 20000

int init(void);
int dinit(void);

void feed_input(int i);
void train(void);
// void compute_cost(int i);

static network_t network;
static const int neurons_per_layer[LAYERS_NUM] = {3, 4, 5, 4};
static float learning_rate;
// float *cost;
// float full_cost;

// each training sample has an array of values, one for each neuron of the input layer
// inputs[TRAINING_EXAMPLES_NUM][INPUT_NEURONS_NUM]
// EX: [[1, 2], [4, 1], [5, 8]]
static float **input;

// each training sample has an array of desired outputs, one for each neuron of the output layer
// desired_outputs[TRAINING_EXAMPLES_NUM][OUTPUT_NEURONS_NUM]
static float **desired_outputs;

// number of training samples
static int num_training_ex;
// int n=1;

int main(void)
{
    // int i;

    // srand(time(0));

    // printf("Enter the number of Layers in Neural Network:\n");
    // scanf("%d",&network.layers_num);

    // num_neurons = (int*) malloc(network.layers_num * sizeof(int));
    // memset(num_neurons,0,network.layers_num *sizeof(int));

    // // Get number of neurons per layer_t
    // for(i=0;i<network.layers_num;i++)
    // {
    //     printf("Enter number of neurons in layer_t[%d]: \n",i+1);
    //     scanf("%d",&num_neurons[i]);
    // }

    // printf("\n");

    // // Initialize the neural network module
    // if(init()!= SUCCESS)
    // {
    //     printf("Error in Initialization...\n");
    //     exit(0);
    // }

    // printf("Enter the learning rate (Usually 0.15): \n");
    // scanf("%f",&learning_rate);
    // printf("\n");

    // printf("Enter the number of training examples: \n");
    // scanf("%d",&num_training_ex);
    // printf("\n");

    // input = (float**) malloc(num_training_ex * sizeof(float*));
    // for(i=0;i<num_training_ex;i++)
    // {
    //     input[i] = (float*)malloc(num_neurons[0] * sizeof(float));
    // }

    // desired_outputs = (float**) malloc(num_training_ex* sizeof(float*));
    // for(i=0;i<num_training_ex;i++)
    // {
    //     desired_outputs[i] = (float*)malloc(num_neurons[network.layers_num-1] * sizeof(float));
    // }

    // cost = (float *) malloc(num_neurons[network.layers_num-1] * sizeof(float));
    // memset(cost,0,num_neurons[network.layers_num-1]*sizeof(float));

    // // Get Training Examples
    // get_inputs();

    // // Get Output Labels
    // get_desired_outputs();

    // train();
    // test_nn();

    // if(dinit()!= SUCCESS)
    // {
    //     printf("Error in Dinitialization...\n");
    // }

    return 0;
}


int init()
{
    if(create_network(LAYERS_NUM, neurons_per_layer) != SUCCESS)
    {
        printf("Error in creating architecture...\n");
        return ERR;
    }

    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS;
}

// Feed inputs to input layer_t
void feed_input(int i)
{
    for(int j=0;j<network.layers[0].neurons_num;j++)
    {
        network.layers[0].neurons[j].actv = input[i][j];
        printf("Input: %f\n",network.layers[0].neurons[j].actv);
    }
}

// Train Neural Network
void train(void)
{
    // Gradient Descent
    for(int it=0;it<ITERATIONS_NUM;it++)
    {
        for(int i=0;i<num_training_ex;i++)
        {
            feed_input(i);
            forward_propagation(&network);
            // compute_cost(i);
            back_propagation(&network, desired_outputs[i]);
            update_weights(&network, learning_rate);
        }
    }
}

// Compute Total Cost
// void compute_cost(int i)
// {
//     float tmpcost=0;
//     float tcost=0;

//     for(int j=0;j<network.layers[network.layers_num-1].neurons_num;j++)
//     {
//         tmpcost = desired_outputs[i][j] - network.layers[network.layers_num-1].neurons[j].actv;
//         cost[j] = (tmpcost * tmpcost)/2;
//         tcost = tcost + cost[j];
//     }   

//     full_cost = (full_cost + tcost)/n;
//     n++;
//     // printf("Full Cost: %f\n",full_cost);
// }


// Test the trained network
// void test_nn(void) 
// {
//     int i;
//     while(1)
//     {
//         printf("Enter input to test:\n");

//         for(i=0;i<network.layers[0].neurons_num;i++)
//         {
//             scanf("%f",&network.layers[0].neurons[i].actv);
//         }
//         forward_prop();
//     }
// }

// TODO: Add different Activation functions
//void activation_functions()

// int dinit(void)
// {
//     // TODO:
//     // Free up all the structures

//     return SUCCESS;
// }