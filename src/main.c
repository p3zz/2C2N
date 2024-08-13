#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utils.h"
#include "network.h"

#define LAYERS_NUM 3
#define ITERATIONS_NUM 100
#define LEARNING_RATE 0.05
#define TRAINING_EXAMPLES_LEN 4

#define INPUT_LEN 2
#define HIDDEN_LEN 4
#define OUTPUT_LEN 1

static network_t network = {0};
static const int neurons_per_layer[LAYERS_NUM] = {INPUT_LEN, HIDDEN_LEN, OUTPUT_LEN};
static const activation_function actv_f[LAYERS_NUM] = {sigmoid, sigmoid, relu};
static const activation_function dactv_f[LAYERS_NUM] = {sigmoid_derivative, sigmoid_derivative, relu_derivative};

// each training sample has an array of values, one for each neuron of the input layer
// inputs[TRAINING_EXAMPLES_NUM][INPUT_NEURONS_NUM]
// EX: [[1, 2], [4, 1], [5, 8]]
static float input[TRAINING_EXAMPLES_LEN][INPUT_LEN] = {0};

// each training sample has an array of desired outputs, one for each neuron of the output layer
// desired_outputs[TRAINING_EXAMPLES_NUM][OUTPUT_NEURONS_NUM]
static float output_targets[TRAINING_EXAMPLES_LEN][OUTPUT_LEN] = {0};

static void initialize_inputs(){
    input[0][0] = 0;
    input[0][1] = 0;

    input[1][0] = 0;
    input[1][1] = 1;

    input[2][0] = 1;
    input[2][1] = 0;

    input[3][0] = 1;
    input[3][1] = 1;

}

static void initialize_outputs(){
    output_targets[0][0] = 0;
    output_targets[1][0] = 0;
    output_targets[2][0] = 0;
    output_targets[3][0] = 1;
}

int main(void)
{
    initialize_inputs();
    initialize_outputs();

    float cost = 0;
    network = create_network(LAYERS_NUM, neurons_per_layer, actv_f, dactv_f);

    for(int i=0;i<network.layers_num;i++){
        printf("[LAYER %d] Neurons: %d\n", i, network.layers[i].neurons_num);
        for(int j=0;j<network.layers[i].neurons_num;j++){
            for(int k=0;k<network.layers[i].neurons[j].weights_num;k++){
                // printf("[LAYER %d NEURON %d]\n", i, j);
                printf("[LAYER %d NEURON %d WEIGHT %d] %.3f\n", i, j, k, network.layers[i].neurons[j].weights[k]);
            }
        }
    }

    for(int it=0;it<ITERATIONS_NUM;it++)
    {
        // printf("Iteration #%d\n", it);
        for(int i=0;i<TRAINING_EXAMPLES_LEN;i++)
        {
            if(train(&network, input[i], INPUT_LEN, output_targets[i], OUTPUT_LEN, LEARNING_RATE, &cost) == ERR){
                return ERR;
            }
            // printf("Cost: %.3f\n", cost);
        }
    }

    for(int i=0;i<TRAINING_EXAMPLES_LEN;i++)
    {
        if(feed_input(&network, input[i], INPUT_LEN) == ERR){
            return ERR;
        }
        forward_propagation(&network);
        float prediction = network.layers[LAYERS_NUM-1].neurons[0].actv;
        float expected_output = output_targets[i][0];
        printf("Input: (%.1f %.1f), Expected output: %.3f, Prediction: %.3f\n", input[i][0], input[i][1], expected_output, prediction);
    }

    destroy_network(&network);

    return SUCCESS;
}
