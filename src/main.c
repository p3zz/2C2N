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

int init(void);
int dinit(void);

int create_architecture(void);
int initialize_weights(void);
void feed_input(int i);
void train_neural_net(void);
void forward_prop(void);
void compute_cost(int i);
void back_prop(int p);
void update_weights(void);
void get_inputs(void);
void get_desired_outputs(void);
void test_nn(void);

int initialize_dummy_weights(void);

static network_t network;
static const int neurons_per_layer[LAYERS_NUM] = {3, 4, 5, 4};
float alpha;
float *cost;
float full_cost;
float **input;
float **desired_outputs;
int num_training_ex;
int n=1;

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
    // scanf("%f",&alpha);
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

    // train_neural_net();
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

//Get Inputs
void  get_inputs()
{
    for(int i=0;i<num_training_ex;i++)
    {
        printf("Enter the Inputs for training example[%d]:\n",i);

        for(int j=0;j<network.layers[0].neurons_num;j++)
        {
            scanf("%f",&input[i][j]);
            
        }
        printf("\n");
    }
}

//Get Labels
void get_desired_outputs()
{
    for(int i=0;i<num_training_ex;i++)
    {
        for(int j=0;j<network.layers[network.layers_num-1].neurons_num;j++)
        {
            printf("Enter the Desired Outputs (Labels) for training example[%d]: \n",i);
            scanf("%f",&desired_outputs[i][j]);
            printf("\n");
        }
    }
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

int initialize_weights(void)
{
    int i,j,k;

    if(network.layers == NULL)
    {
        printf("No network.layers in Neural Network...\n");
        return ERR;
    }

    printf("Initializing weights...\n");

    for(i=0;i<network.layers_num-1;i++)
    {
        
        for(j=0;j<network.layers[i].neurons_num;j++)
        {
            for(k=0;k<network.layers[i+1].neurons_num;k++)
            {
                // Initialize Output Weights for each neuron
                network.layers[i].neurons[j].weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, network.layers[i].neurons[j].weights[k]);
                network.layers[i].neurons[j].dw[k] = 0.0;
            }

            if(i>0) 
            {
                network.layers[i].neurons[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }   
    printf("\n");
    
    for (j=0; j<network.layers[network.layers_num-1].neurons_num; j++)
    {
        network.layers[network.layers_num-1].neurons[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS;
}

// Train Neural Network
void train_neural_net(void)
{
    int i;
    int it=0;

    // Gradient Descent
    for(it=0;it<20000;it++)
    {
        for(i=0;i<num_training_ex;i++)
        {
            feed_input(i);
            forward_prop();
            compute_cost(i);
            back_prop(i);
            update_weights();
        }
    }
}

void update_weights(void)
{
    // for each layer (excluding the output layer)
    for(int i=0;i<network.layers_num-1;i++)
    {
        // for each neuron of the current layer
        for(int j=0;j<network.layers[i].neurons_num;j++)
        {
            neuron_t* neuron = &network.layers[i].neurons[j];
            // for each neuron of the next layer
            for(int k=0;k<network.layers[i+1].neurons_num;k++)
            {
                // Update Weights
                neuron->weights[k] = update_weight(neuron->weights[k], alpha, neuron->dw[k]);
            }
            
            // Update Bias
            neuron->bias = update_bias(neuron->bias, alpha,  neuron->dbias);
        }
    }   
}

void forward_prop(void)
{
    // for each layer of the network
    // start from the second layer so we skip the input one
    for(int i=1;i<network.layers_num;i++)
    {   
        neuron_t* current_neurons = network.layers[i].neurons;
        neuron_t* previous_neurons = network.layers[i-1].neurons;

        // for each neuron of the current layer
        for(int j=0;j<network.layers[i].neurons_num;j++)
        {
            current_neurons[j].z = current_neurons[j].bias;

            // for each neuron of the previous layer
            for(int k=0;k<network.layers[i-1].neurons_num;k++)
            {
                // update the output of each neuron of the current layer
                current_neurons[j].z  = update_output(previous_neurons[k].actv, previous_neurons[k].weights[j], current_neurons[j].z);
            }

            // then apply to the output a given activation function based on the type of current layer
            // the last layer is the output layer, the previous ones are hidden layers because we are starting from the second layer (layer[1])
            // the first layer is the input layer
            bool is_hidden = i < (network.layers_num - 1);
            // Relu Activation Function for Hidden Layers
            if(is_hidden)
            {
                current_neurons[j].actv = relu(current_neurons[j].z);
            }
            
            // Sigmoid Activation function for Output Layer
            else
            {
                current_neurons[j].actv = sigmoid(current_neurons[j].z);
            }
            
            // print the output of the neuron
            printf("Output: %d\n", (int)round(network.layers[i].neurons[j].actv));
            printf("\n");
        }
    }
}

// Compute Total Cost
void compute_cost(int i)
{
    float tmpcost=0;
    float tcost=0;

    for(int j=0;j<network.layers[network.layers_num-1].neurons_num;j++)
    {
        tmpcost = desired_outputs[i][j] - network.layers[network.layers_num-1].neurons[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
    }   

    full_cost = (full_cost + tcost)/n;
    n++;
    // printf("Full Cost: %f\n",full_cost);
}

// Back Propogate Error
void back_prop(int p)
{
    // Output Layer
    // for each neuron of the output layer
    for(int j=0;j<network.layers[network.layers_num-1].neurons_num;j++)
    {           
        // compute delta error
        neuron_t* current_neuron = &(network.layers[network.layers_num-1].neurons[j]);
        current_neuron->dz = (current_neuron->actv - desired_outputs[p][j]) * sigmoid_derivative(current_neuron->actv);

        // for each neuron of the previous layer (the one before the output layer)
        for(int k=0;k<network.layers[network.layers_num-2].neurons_num;k++)
        {   
            neuron_t* previous_neuron = &(network.layers[network.layers_num-2].neurons[k]);
            // compute the delta weight and the delta output 
            previous_neuron->dw[j] = (current_neuron->dz * previous_neuron->actv);
            previous_neuron->dactv = previous_neuron->weights[j] * current_neuron->dz;
        }

        current_neuron->dbias = current_neuron->dz;
    }

    // Hidden Layers
    // for each hidden layer iterate backwards from the last hidden layer to the first
    for(int i=network.layers_num-2;i>0;i--)
    {
        // for each neuron of the current layer
        for(int j=0;j<network.layers[i].neurons_num;j++)
        {
            neuron_t* current_neuron = &(network.layers[i].neurons[j]);
            // compute the partial derivative w.r.t. output
            current_neuron->dz = relu_derivative(current_neuron->dactv);

            // for each neuron of the previous layer
            for(int k=0;k<network.layers[i-1].neurons_num;k++)
            {
                neuron_t* previous_neuron = &(network.layers[i-1].neurons[k]);
                // compute the derivative w.r.t. weight
                previous_neuron->dw[j] = current_neuron->dz * previous_neuron->actv;
                
                // if the current layer is not the input layer
                if(i>1)
                {
                    // compute the derivative w.r.t. output
                    previous_neuron->dactv = previous_neuron->weights[j] * current_neuron->dz;
                }
            }
            // compute the derivative w.r.t. bias
            current_neuron->dbias = current_neuron->dz;
        }
    }
}

// Test the trained network
void test_nn(void) 
{
    int i;
    while(1)
    {
        printf("Enter input to test:\n");

        for(i=0;i<network.layers[0].neurons_num;i++)
        {
            scanf("%f",&network.layers[0].neurons[i].actv);
        }
        forward_prop();
    }
}

// TODO: Add different Activation functions
//void activation_functions()

int dinit(void)
{
    // TODO:
    // Free up all the structures

    return SUCCESS;
}