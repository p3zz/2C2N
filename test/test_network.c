#include <unity.h>
#include "stdbool.h"
#include "network.h"

void setUp()
{

}

void tearDown()
{

}

void test_always_true(void){
    TEST_ASSERT_TRUE(true);
}

void test_forward_propagation_1(void){
    const int input_neurons_num = 2;
    const int hidden_neurons_num = 3;
    const int output_neurons_num = 2;
    const int layers_num = 4;
    const int neurons_per_layer[4] = {input_neurons_num, hidden_neurons_num, hidden_neurons_num, output_neurons_num};
    const float input[2] = {1.f, 1.f};
    const float output_targets[2] = {1.f, 0.f};
    const float learning_rate = 0.15f;
    static const activation_function actv_f[4] = {relu, relu, relu, relu};
    static const activation_function dactv_f[4] = {relu_derivative, relu_derivative, relu_derivative, relu_derivative};

    network_t network = create_network(layers_num, neurons_per_layer, actv_f, dactv_f);
    // setup layer 0
    // setup neuron 0
    network.layers[0].neurons[0].weights[0] = 0.3;
    network.layers[0].neurons[0].weights[1] = 0.4;
    network.layers[0].neurons[0].weights[2] = 0.7;
    // setup neuron 1
    network.layers[0].neurons[1].weights[0] = 0.1;
    network.layers[0].neurons[1].weights[1] = 0.5;
    network.layers[0].neurons[1].weights[2] = 0.6;

    // setup layer 1
    // setup neuron 0
    network.layers[1].neurons[0].bias = 0.2;
    network.layers[1].neurons[0].weights[0] = 0.3;
    network.layers[1].neurons[0].weights[1] = 0.2;
    network.layers[1].neurons[0].weights[2] = 0.3;
    // setup neuron 1
    network.layers[1].neurons[1].bias = 0.2;
    network.layers[1].neurons[1].weights[0] = 0.3;
    network.layers[1].neurons[1].weights[1] = 0.2;
    network.layers[1].neurons[1].weights[2] = 0.1;
    // setup neuron 2
    network.layers[1].neurons[2].bias = 0.2;
    network.layers[1].neurons[2].weights[0] = 0.6;
    network.layers[1].neurons[2].weights[1] = 0.7;
    network.layers[1].neurons[2].weights[2] = 0.9;

    // setup layer 2
    // setup neuron 0
    network.layers[2].neurons[0].bias = 0.3;
    network.layers[2].neurons[0].weights[0] = 0.4;
    network.layers[2].neurons[0].weights[1] = 0.8;
    // setup neuron 1
    network.layers[2].neurons[1].bias = 0.4;
    network.layers[2].neurons[1].weights[0] = 0.6;
    network.layers[2].neurons[1].weights[1] = 0.7;
    // setup neuron 2
    network.layers[2].neurons[2].bias = 0.4;
    network.layers[2].neurons[2].weights[0] = 0.1;
    network.layers[2].neurons[2].weights[1] = 0.2;

    network.layers[3].neurons[0].bias = 0.3;
    network.layers[3].neurons[1].bias = 0.4;

    int res = feed_input(&network, input, 2);
    forward_propagation(&network);
    // layer 0
    // neuron 0
    TEST_ASSERT_EQUAL_FLOAT(1.f, network.layers[0].neurons[0].actv);
    // neuron 1
    TEST_ASSERT_EQUAL_FLOAT(1.f, network.layers[0].neurons[1].actv);
    
    // layer 1
    // neuron 0
    TEST_ASSERT_EQUAL_FLOAT(0.6f, network.layers[1].neurons[0].z);
    TEST_ASSERT_EQUAL_FLOAT(0.6f, network.layers[1].neurons[0].actv);
    // neuron 1
    TEST_ASSERT_EQUAL_FLOAT(1.1f, network.layers[1].neurons[1].z);
    TEST_ASSERT_EQUAL_FLOAT(1.1f, network.layers[1].neurons[1].actv);
    // neuron 2
    TEST_ASSERT_EQUAL_FLOAT(1.5f, network.layers[1].neurons[2].z);
    TEST_ASSERT_EQUAL_FLOAT(1.5f, network.layers[1].neurons[2].actv);

    // layer 2
    // neuron 0
    TEST_ASSERT_EQUAL_FLOAT(1.71f, network.layers[2].neurons[0].z);
    TEST_ASSERT_EQUAL_FLOAT(1.71f, network.layers[2].neurons[0].actv);
    // neuron 1
    TEST_ASSERT_EQUAL_FLOAT(1.79, network.layers[2].neurons[1].z);
    TEST_ASSERT_EQUAL_FLOAT(1.79, network.layers[2].neurons[1].actv);
    // neuron 2
    TEST_ASSERT_EQUAL_FLOAT(2.04, network.layers[2].neurons[2].z);
    TEST_ASSERT_EQUAL_FLOAT(2.04, network.layers[2].neurons[2].actv);

    // layer 3
    // neuron 0
    TEST_ASSERT_EQUAL_FLOAT(2.262f, network.layers[3].neurons[0].z);
    TEST_ASSERT_EQUAL_FLOAT(2.262f, network.layers[3].neurons[0].actv);
    // neuron 1
    TEST_ASSERT_EQUAL_FLOAT(3.429f, network.layers[3].neurons[1].z);
    TEST_ASSERT_EQUAL_FLOAT(3.429f, network.layers[3].neurons[1].actv);
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_forward_propagation_1);

    int result = UNITY_END();

    return result;
}