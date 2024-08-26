#include <unity.h>
#include <layer.h>
#include "stdbool.h"
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

void test_init_conv_layer(void){
    conv_layer_t layer = {0};
    init_conv_layer(&layer, 3, 2, 1, 1, 0, ACTIVATION_TYPE_RELU);
    TEST_ASSERT_EQUAL_INT(0, layer.padding);
    TEST_ASSERT_EQUAL_INT(1, layer.stride);
    TEST_ASSERT_EQUAL_INT(1, layer.kernels_n);
    TEST_ASSERT_EQUAL_INT(2, layer.kernels[0].depth);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[0].cols_n);
    TEST_ASSERT_EQUAL_FLOAT(0.8401877, layer.kernels[0].layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(0.3943829, layer.kernels[0].layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_INT(3 ,layer.kernels[0].layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[1].cols_n);
    TEST_ASSERT_EQUAL_FLOAT(0.55397, layer.kernels[0].layers[1].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(0.6288709, layer.kernels[0].layers[1].values[0][2]);
    destroy_conv_layer(&layer);
}

void test_process_conv_layer(void){
    conv_layer_t layer = {0};
    init_conv_layer(&layer, 2, 2, 2, 1, 0, ACTIVATION_TYPE_RELU);
    const float input_vals[2][3][3] = {
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        },
        {
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1},
        }
    };
    matrix3d_t input = {0};
    create_matrix3d(&input, 3, 3, 2);
    for(int i=0;i<input.depth;i++){
        for(int j=0;j<input.layers[i].rows_n;j++){
            for(int k=0;k<input.layers[i].cols_n;k++){
                input.layers[i].values[j][k] = input_vals[i][j][k];
            }
        }
    }

    process_conv_layer(&layer, &input);
    // printf("Input\n");
    // matrix3d_print(&input);
    // printf("Kernels\n");
    // matrix3d_print(&layer.kernels[0]);
    // matrix3d_print(&layer.kernels[1]);
    // printf("Output\n");
    // matrix3d_print(&output);
    // TODO add output values check
    TEST_ASSERT_EQUAL_INT(2, layer.output.depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].cols_n);
    TEST_ASSERT_EQUAL_FLOAT(13.55092, layer.output.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(13.74786, layer.output.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(13.13456, layer.output.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(12.19739, layer.output.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(22.64161, layer.output.layers[1].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(22.78511, layer.output.layers[1].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(21.07566, layer.output.layers[1].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(19.77494, layer.output.layers[1].values[1][1]);

    destroy_conv_layer(&layer);
    destroy_matrix3d(&input);
}

void test_process_pool_layer_average(void){
    pool_layer_t layer = {0};
    init_pool_layer(&layer, 2, 0, 1, POOLING_TYPE_AVERAGE);
    const float input_vals[2][3][3] = {
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        },
        {
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1},
        }
    };
    matrix3d_t input = {0};
    create_matrix3d(&input, 3, 3, 2);
    for(int i=0;i<input.depth;i++){
        for(int j=0;j<input.layers[i].rows_n;j++){
            for(int k=0;k<input.layers[i].cols_n;k++){
                input.layers[i].values[j][k] = input_vals[i][j][k];
            }
        }
    }

    process_pool_layer(&layer, &input);
    // printf("Input\n");
    // matrix3d_print(&input);
    // printf("Output\n");
    // matrix3d_print(&output);
    TEST_ASSERT_EQUAL_INT(2, layer.output.depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].cols_n);
    TEST_ASSERT_EQUAL_FLOAT(3, layer.output.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(4, layer.output.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(6, layer.output.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(7, layer.output.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(7, layer.output.layers[1].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(6, layer.output.layers[1].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(4, layer.output.layers[1].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(3, layer.output.layers[1].values[1][1]);
    
    destroy_matrix3d(&input);
    destroy_pool_layer(&layer);
}

void test_process_pool_layer_max(void){
    pool_layer_t layer = {0};
    init_pool_layer(&layer, 2, 0, 1, POOLING_TYPE_MAX);
    const float input_vals[2][3][3] = {
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        },
        {
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1},
        }
    };
    matrix3d_t input = {0};
    create_matrix3d(&input, 3, 3, 2);
    for(int i=0;i<input.depth;i++){
        for(int j=0;j<input.layers[i].rows_n;j++){
            for(int k=0;k<input.layers[i].cols_n;k++){
                input.layers[i].values[j][k] = input_vals[i][j][k];
            }
        }
    }

    process_pool_layer(&layer, &input);
    // printf("Input\n");
    // matrix3d_print(&input);
    // printf("Output\n");
    // matrix3d_print(&output);
    TEST_ASSERT_EQUAL_INT(2, layer.output.depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].cols_n);
    TEST_ASSERT_EQUAL_FLOAT(5, layer.output.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(6, layer.output.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(8, layer.output.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(9, layer.output.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(9, layer.output.layers[1].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(8, layer.output.layers[1].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(6, layer.output.layers[1].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(5, layer.output.layers[1].values[1][1]);
    destroy_pool_layer(&layer);
    destroy_matrix3d(&input);
}

void test_process_dense_layer(void){
    dense_layer_t layer = {0};
    init_dense_layer(&layer, 4, 2, ACTIVATION_TYPE_RELU);

    const float input_vals[4] = {3.f, 4.f, 2.f, 1.f};
    matrix2d_t input = {0};
    create_matrix2d(&input, 1, 4);
    for(int j=0;j<input.cols_n;j++){
        input.values[0][j] = input_vals[j];
    }
    feed_dense_layer(&layer, &input);
    matrix2d_print(&layer.inputs);
    matrix2d_print(&layer.weights);
    matrix2d_print(&layer.biases);
    
    process_dense_layer(&layer);
    TEST_ASSERT_EQUAL_INT(1, layer.output.rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(5.76589, layer.output.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(6.480252, layer.output.values[0][1]);
    destroy_dense_layer(&layer);
    destroy_matrix2d(&input);
}

void test_backpropagation_dense_layer(void){
    dense_layer_t layer = {0};
    init_dense_layer(&layer, 3, 2, ACTIVATION_TYPE_RELU);

    const float input_vals[3] = {1.71f, 1.79f, 2.04f};
    const float output_targets[2] = {1.f, 0.f};

    matrix2d_t input = {0};
    create_matrix2d(&input, 1, 3);
    for(int j=0;j<input.cols_n;j++){
        input.values[0][j] = input_vals[j];
    }

    matrix2d_t output_target = {0};
    create_matrix2d(&output_target, 1, 2);
    for(int j=0;j<output_target.cols_n;j++){
        output_target.values[0][j] = output_targets[j];
    }

    layer.weights.values[0][0] = 0.4;
    layer.weights.values[0][1] = 0.8;

    layer.weights.values[1][0] = 0.6;
    layer.weights.values[1][1] = 0.7;

    layer.weights.values[2][0] = 0.1;
    layer.weights.values[2][1] = 0.2;

    layer.biases.values[0][0] = 0.3;
    layer.biases.values[0][1] = 0.4;

    feed_dense_layer(&layer, &input);
    process_dense_layer(&layer);

    TEST_ASSERT_EQUAL_FLOAT(2.262f, layer.output.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.429f, layer.output.values[0][1]);

    matrix2d_t d_input = {0};
    // create_matrix2d(&d_input, 1, 2);

    compute_cost_derivative(&layer.output_activated, &output_target, &d_input);

    TEST_ASSERT_EQUAL_FLOAT(2.524f, d_input.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(6.858f, d_input.values[0][1]);

    backpropagation_dense_layer(&layer, &d_input, 0.15f);

    TEST_ASSERT_EQUAL_FLOAT(6.496, layer.d_inputs.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(6.315, layer.d_inputs.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(1.624, layer.d_inputs.values[0][2]);

    TEST_ASSERT_EQUAL_FLOAT(-0.2474061, layer.weights.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(-0.9590769, layer.weights.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(-0.077694, layer.weights.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(-1.141373, layer.weights.values[1][1]);

    destroy_dense_layer(&layer);
    destroy_matrix2d(&input);
    destroy_matrix2d(&output_target);
    destroy_matrix2d(&d_input);
}

void test_perceptron_or(void){
    const float learning_rate = 0.05f;
    const int iterations_n = 10000;

    matrix3d_t inputs = {0};
    create_matrix3d(&inputs, 1, 2, 4);
    inputs.layers[0].values[0][0] = 0;
    inputs.layers[0].values[0][1] = 0;

    inputs.layers[1].values[0][0] = 0;
    inputs.layers[1].values[0][1] = 1;

    inputs.layers[2].values[0][0] = 1;
    inputs.layers[2].values[0][1] = 0;

    inputs.layers[3].values[0][0] = 1;
    inputs.layers[3].values[0][1] = 1;

    matrix3d_t output_targets = {0};
    create_matrix3d(&output_targets, 1, 1, 4);
    output_targets.layers[0].values[0][0] = 0;

    output_targets.layers[1].values[0][0] = 1;

    output_targets.layers[2].values[0][0] = 1;

    output_targets.layers[3].values[0][0] = 1;
    
    dense_layer_t input_layer = {0};
    init_dense_layer(&input_layer, 2, 4, ACTIVATION_TYPE_RELU);
    
    dense_layer_t hidden_layer = {0};
    init_dense_layer(&hidden_layer, 4, 1, ACTIVATION_TYPE_RELU);

    matrix2d_t d_input = {0};

    for(int i=0;i<iterations_n;i++){
        for(int j=0;j<inputs.depth;j++){
            feed_dense_layer(&input_layer, &inputs.layers[j]);
            process_dense_layer(&input_layer);
            feed_dense_layer(&hidden_layer, &input_layer.output_activated);
            process_dense_layer(&hidden_layer);
            compute_cost_derivative(&hidden_layer.output_activated, &output_targets.layers[j], &d_input);
            backpropagation_dense_layer(&hidden_layer, &d_input, learning_rate);
            backpropagation_dense_layer(&input_layer, &d_input, learning_rate);
            destroy_matrix2d(&d_input);
        }
    }

    TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.rows_n);
    TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.cols_n);

    for(int i=0;i<inputs.depth;i++){
        feed_dense_layer(&input_layer, &inputs.layers[i]);
        process_dense_layer(&input_layer);
        feed_dense_layer(&hidden_layer, &input_layer.output_activated);
        process_dense_layer(&hidden_layer);
        TEST_ASSERT_FLOAT_WITHIN(0.00001, output_targets.layers[i].values[0][0], hidden_layer.output_activated.values[0][0]);
    }

    destroy_matrix3d(&inputs);
    destroy_matrix3d(&output_targets);
    destroy_dense_layer(&input_layer);
    destroy_dense_layer(&hidden_layer);
}

int main(void)
{
    srand(0);
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_init_conv_layer);
    RUN_TEST(test_process_conv_layer);
    RUN_TEST(test_process_pool_layer_average);
    RUN_TEST(test_process_pool_layer_max);
    RUN_TEST(test_process_dense_layer);
    RUN_TEST(test_backpropagation_dense_layer);
    RUN_TEST(test_perceptron_or);
    int result = UNITY_END();

    return result;
}