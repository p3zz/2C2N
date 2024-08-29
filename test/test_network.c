#include <unity.h>
#include "stdbool.h"
#include "network.h"
#include "stdlib.h"

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
    TEST_ASSERT_EQUAL_INT(SUCCESS, res);
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

    res = back_propagation(&network, output_targets, 2);
    TEST_ASSERT_EQUAL_INT(SUCCESS, res);

    // layer 3
    // neuron 0
    TEST_ASSERT_EQUAL_FLOAT(2.524f, network.layers[3].neurons[0].dactv);
    TEST_ASSERT_EQUAL_FLOAT(2.524f, network.layers[3].neurons[0].dbias);
    // neuron 1
    TEST_ASSERT_EQUAL_FLOAT(6.858f, network.layers[3].neurons[1].dactv);
    TEST_ASSERT_EQUAL_FLOAT(6.858f, network.layers[3].neurons[1].dbias);

    // layer 2
    // neuron 0
    TEST_ASSERT_EQUAL_FLOAT(4.316041, network.layers[2].neurons[0].dw[0]);
    TEST_ASSERT_EQUAL_FLOAT(11.72718, network.layers[2].neurons[0].dw[1]);
    TEST_ASSERT_EQUAL_FLOAT(6.496, network.layers[2].neurons[0].dactv);

    // neuron 1
    TEST_ASSERT_EQUAL_FLOAT(4.51796, network.layers[2].neurons[1].dw[0]);
    TEST_ASSERT_EQUAL_FLOAT(12.27582, network.layers[2].neurons[1].dw[1]);
    TEST_ASSERT_EQUAL_FLOAT(6.315, network.layers[2].neurons[1].dactv);

    // neuron 2
    TEST_ASSERT_EQUAL_FLOAT(5.14896, network.layers[2].neurons[2].dw[0]);
    TEST_ASSERT_EQUAL_FLOAT(13.99032, network.layers[2].neurons[2].dw[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.624, network.layers[2].neurons[2].dactv);

    gradient_descents(&network, 0.15f);
    TEST_ASSERT_EQUAL_FLOAT(-0.2474061, network.layers[2].neurons[0].weights[0]);
    TEST_ASSERT_EQUAL_FLOAT(-0.9590769, network.layers[2].neurons[0].weights[1]);
    TEST_ASSERT_EQUAL_FLOAT(-0.077694, network.layers[2].neurons[1].weights[0]);
    TEST_ASSERT_EQUAL_FLOAT(-1.141373, network.layers[2].neurons[1].weights[1]);
}

int main(void)
{
    srand(0);
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_forward_propagation_1);

    int result = UNITY_END();

    return result;
}