#include <unity.h>
#include <layer.h>
#include "stdbool.h"
#include "stdlib.h"
#include "stdio.h"

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
    const int input_height = 10;
    const int input_width = 10;
    const int input_depth = 2;
    const int kernel_size = 3;
    const int kernels_n = 1;
    const int stride = 1;
    const int padding = 0;

    conv_layer_t layer = {0};
    init_conv_layer(&layer, input_height, input_width, input_depth, kernel_size, kernels_n, stride, padding, ACTIVATION_TYPE_RELU);
    TEST_ASSERT_EQUAL_INT(0, layer.padding);
    TEST_ASSERT_EQUAL_INT(1, layer.stride);
    // check kernels
    TEST_ASSERT_EQUAL_INT(1, layer.kernels_n);
    TEST_ASSERT_EQUAL_INT(2, layer.kernels[0].depth);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(3 ,layer.kernels[0].layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[1].cols_n);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[1].cols_n);
    TEST_ASSERT_EQUAL_INT(3, layer.kernels[0].layers[1].cols_n);
    // check input
    TEST_ASSERT_EQUAL_INT(2, layer.input.depth);
    TEST_ASSERT_EQUAL_INT(10, layer.input.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(10, layer.input.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(10, layer.input.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(10, layer.input.layers[1].cols_n);
    // check d_input
    TEST_ASSERT_EQUAL_INT(2, layer.d_input.depth);
    TEST_ASSERT_EQUAL_INT(10, layer.d_input.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(10, layer.d_input.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(10, layer.d_input.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(10, layer.d_input.layers[1].cols_n);
    // check biases
    TEST_ASSERT_EQUAL_INT(8, layer.biases[0].rows_n);
    TEST_ASSERT_EQUAL_INT(8, layer.biases[0].rows_n);
    // check output
    TEST_ASSERT_EQUAL_INT(1, layer.output.depth);
    TEST_ASSERT_EQUAL_INT(8, layer.output.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(8, layer.output.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(1, layer.output_activated.depth);
    TEST_ASSERT_EQUAL_INT(8, layer.output_activated.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(8, layer.output_activated.layers[0].cols_n);
    destroy_conv_layer(&layer);
}

void test_process_conv_layer(void){
    const int input_height = 3;
    const int input_width = 3;
    const int input_depth = 2;
    const int kernel_size = 2;
    const int kernels_n = 2;
    const int stride = 1;
    const int padding = 0;
    conv_layer_t layer = {0};
    init_conv_layer(&layer, input_height, input_width, input_depth, kernel_size, kernels_n, stride, padding, ACTIVATION_TYPE_RELU);
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
    create_matrix3d(&input, input_height, input_width, input_depth);
    for(int i=0;i<input.depth;i++){
        for(int j=0;j<input.layers[i].rows_n;j++){
            for(int k=0;k<input.layers[i].cols_n;k++){
                input.layers[i].values[j][k] = input_vals[i][j][k];
            }
        }
    }

    layer.kernels[0].layers[0].values[0][0] = 0.142;
    layer.kernels[0].layers[0].values[0][1] = 0.607;
    layer.kernels[0].layers[0].values[1][0] = 0.016;
    layer.kernels[0].layers[0].values[1][1] = 0.243;
    
    layer.kernels[0].layers[1].values[0][0] = 0.137;
    layer.kernels[0].layers[1].values[0][1] = 0.804;
    layer.kernels[0].layers[1].values[1][0] = 0.157;
    layer.kernels[0].layers[1].values[1][1] = 0.401;

    layer.kernels[1].layers[0].values[0][0] = 0.130;
    layer.kernels[1].layers[0].values[0][1] = 0.109;
    layer.kernels[1].layers[0].values[1][0] = 0.999;
    layer.kernels[1].layers[0].values[1][1] = 0.218;
    
    layer.kernels[1].layers[1].values[0][0] = 0.513;
    layer.kernels[1].layers[1].values[0][1] = 0.839;
    layer.kernels[1].layers[1].values[1][0] = 0.613;
    layer.kernels[1].layers[1].values[1][1] = 0.296;

    layer.biases[0].values[0][0] = 0.398;
    layer.biases[0].values[0][1] = 0.815;
    layer.biases[0].values[1][0] = 0.684;
    layer.biases[0].values[1][1] = 0.911;

    layer.biases[1].values[0][0] = 0.556;
    layer.biases[1].values[0][1] = 0.417;
    layer.biases[1].values[1][0] = 0.170;
    layer.biases[1].values[1][1] = 0.613;

    feed_conv_layer(&layer, &input);
    process_conv_layer(&layer);
    printf("Kernels\n");
    matrix3d_print(&layer.kernels[0]);
    matrix3d_print(&layer.kernels[1]);
    printf("Biases\n");
    matrix2d_print(&layer.biases[0]);
    matrix2d_print(&layer.biases[1]);
    printf("Output\n");
    matrix3d_print(&layer.output);
    matrix3d_print(&layer.output_activated);

    TEST_ASSERT_EQUAL_INT(2, layer.output.depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.layers[1].cols_n);
    TEST_ASSERT_EQUAL_FLOAT(13.645, layer.output.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(13.571, layer.output.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(12.458, layer.output.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(12.194, layer.output.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(22.477, layer.output.layers[1].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(21.533, layer.output.layers[1].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(19.676, layer.output.layers[1].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(19.314, layer.output.layers[1].values[1][1]);

    destroy_conv_layer(&layer);
    destroy_matrix3d(&input);
}

void test_process_pool_layer_average(void){
    pool_layer_t layer = {0};
    init_pool_layer(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_AVERAGE);
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

    feed_pool_layer(&layer, &input);
    process_pool_layer(&layer);
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
    init_pool_layer(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_MAX);
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

    feed_pool_layer(&layer, &input);
    process_pool_layer(&layer);
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
    layer.weights.values[0][0] = 0.294;
    layer.weights.values[0][1] = 0.232;
    layer.weights.values[1][0] = 0.584;
    layer.weights.values[1][1] = 0.244;
    layer.weights.values[2][0] = 0.152;
    layer.weights.values[2][1] = 0.732;
    layer.weights.values[3][0] = 0.125;
    layer.weights.values[3][1] = 0.793;

    layer.biases.values[0][0] = 0.164;
    layer.biases.values[0][1] = 0.745;

    feed_dense_layer(&layer, &input);
    
    process_dense_layer(&layer);
    TEST_ASSERT_EQUAL_INT(1, layer.output.rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(3.811, layer.output.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(4.674, layer.output.values[0][1]);
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
    create_matrix2d(&d_input, output_target.rows_n, output_target.cols_n);

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

void test_backpropagation_conv_layer(void){
    const int input_height = 3;
    const int input_width = 3;
    const int input_depth = 2;
    const int kernel_size = 2;
    const int kernels_n = 2;
    const int stride = 1;
    const int padding = 0;
    const float learning_rate = 0.05f;
    conv_layer_t layer = {0};
    init_conv_layer(&layer, input_height, input_width, input_depth, kernel_size, kernels_n, stride, padding, ACTIVATION_TYPE_RELU);
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

    layer.kernels[0].layers[0].values[0][0] = 0.985;
    layer.kernels[0].layers[0].values[0][1] = 0.935;
    layer.kernels[0].layers[0].values[1][0] = 0.684;
    layer.kernels[0].layers[0].values[1][1] = 0.383;

    layer.kernels[0].layers[1].values[0][0] = 0.750;
    layer.kernels[0].layers[1].values[0][1] = 0.369;
    layer.kernels[0].layers[1].values[1][0] = 0.294;
    layer.kernels[0].layers[1].values[1][1] = 0.232;

    layer.kernels[1].layers[0].values[0][0] = 0.584;
    layer.kernels[1].layers[0].values[0][1] = 0.244;
    layer.kernels[1].layers[0].values[1][0] = 0.152;
    layer.kernels[1].layers[0].values[1][1] = 0.732;

    layer.kernels[1].layers[1].values[0][0] = 0.125;
    layer.kernels[1].layers[1].values[0][1] = 0.793;
    layer.kernels[1].layers[1].values[1][0] = 0.164;
    layer.kernels[1].layers[1].values[1][1] = 0.745;

    layer.biases[0].values[0][0] = 0.075;
    layer.biases[0].values[0][1] = 0.950;
    layer.biases[0].values[1][0] = 0.053;
    layer.biases[0].values[1][1] = 0.522;

    layer.biases[1].values[0][0] = 0.176;
    layer.biases[1].values[0][1] = 0.240;
    layer.biases[1].values[1][0] = 0.798;
    layer.biases[1].values[1][1] = 0.733;

    feed_conv_layer(&layer, &input);
    process_conv_layer(&layer);

    matrix3d_t output_targets = {0};
    // the layer produces an output of 2x2x2
    create_matrix3d(&output_targets, 2, 2, 2);
    output_targets.layers[0].values[0][0] = 1;
    output_targets.layers[0].values[0][1] = 1;
    output_targets.layers[0].values[1][0] = 1;
    output_targets.layers[0].values[1][1] = 1;
    output_targets.layers[1].values[0][0] = 1;
    output_targets.layers[1].values[0][1] = 1;
    output_targets.layers[1].values[1][0] = 1;
    output_targets.layers[1].values[1][1] = 1;
    
    matrix3d_t d_input = {0};
    create_matrix3d(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, layer.output_activated.depth);
    for(int i=0;i<d_input.depth;i++){
        compute_cost_derivative(&layer.output_activated.layers[i], &output_targets.layers[i], &d_input.layers[i]);
    }
    TEST_ASSERT_EQUAL_INT(2, d_input.depth);
    TEST_ASSERT_EQUAL_INT(2, d_input.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, d_input.layers[0].cols_n);
    printf("[before] Kernel\n");
    matrix3d_print(&layer.kernels[0]);
    matrix3d_print(&layer.kernels[1]);
    printf("[before] bias\n");
    matrix2d_print(&layer.biases[0]);
    matrix2d_print(&layer.biases[1]);
    backpropagation_conv_layer(&layer, &d_input, learning_rate);
    printf("[after] Kernel\n");
    matrix3d_print(&layer.kernels[0]);
    matrix3d_print(&layer.kernels[1]);
    printf("[after] bias\n");
    matrix2d_print(&layer.biases[0]);
    matrix2d_print(&layer.biases[1]);
    printf("[after] d_input\n");
    matrix3d_print(&layer.d_input);

    destroy_matrix3d(&input);
    destroy_matrix3d(&output_targets);
    destroy_matrix3d(&d_input);
    destroy_conv_layer(&layer);
    // TEST_ASSERT_TRUE(false);
}

void test_backpropagation_max_pool_layer(void){
    pool_layer_t layer = {0};
    init_pool_layer(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_MAX);
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

    feed_pool_layer(&layer, &input);
    process_pool_layer(&layer);

    matrix3d_t output_targets = {0};
    // the layer produces an output of 2x2x2
    create_matrix3d(&output_targets, 2, 2, 2);
    output_targets.layers[0].values[0][0] = 1;
    output_targets.layers[0].values[0][1] = 1;
    output_targets.layers[0].values[1][0] = 1;
    output_targets.layers[0].values[1][1] = 1;
    output_targets.layers[1].values[0][0] = 1;
    output_targets.layers[1].values[0][1] = 1;
    output_targets.layers[1].values[1][0] = 1;
    output_targets.layers[1].values[1][1] = 1;

    matrix3d_t d_input = {0};
    create_matrix3d(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, layer.output.depth);
    for(int i=0;i<d_input.depth;i++){
        compute_cost_derivative(&layer.output.layers[i], &output_targets.layers[i], &d_input.layers[i]);
    }

    backpropagation_pool_layer(&layer, &d_input);

    TEST_ASSERT_EQUAL_INT(2, layer.d_input.depth);
    TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].cols_n);

    TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[0][2]);

    TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0, layer.d_input.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(10.0, layer.d_input.layers[0].values[1][2]);

    TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[2][0]);
    TEST_ASSERT_EQUAL_FLOAT(14.0, layer.d_input.layers[0].values[2][1]);
    TEST_ASSERT_EQUAL_FLOAT(16.0, layer.d_input.layers[0].values[2][2]);

    destroy_pool_layer(&layer);
    destroy_matrix3d(&d_input);
    destroy_matrix3d(&input);
    destroy_matrix3d(&output_targets);
}

void test_backpropagation_avg_pool_layer(void){
    pool_layer_t layer = {0};
    init_pool_layer(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_AVERAGE);
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

    feed_pool_layer(&layer, &input);
    process_pool_layer(&layer);

    matrix3d_t output_targets = {0};
    // the layer produces an output of 2x2x2
    create_matrix3d(&output_targets, 2, 2, 2);
    output_targets.layers[0].values[0][0] = 1;
    output_targets.layers[0].values[0][1] = 1;
    output_targets.layers[0].values[1][0] = 1;
    output_targets.layers[0].values[1][1] = 1;
    output_targets.layers[1].values[0][0] = 1;
    output_targets.layers[1].values[0][1] = 1;
    output_targets.layers[1].values[1][0] = 1;
    output_targets.layers[1].values[1][1] = 1;

    matrix3d_t d_input = {0};
    create_matrix3d(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, layer.output.depth);
    for(int i=0;i<d_input.depth;i++){
        compute_cost_derivative(&layer.output.layers[i], &output_targets.layers[i], &d_input.layers[i]);
    }

    TEST_ASSERT_EQUAL_FLOAT(4.0, d_input.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0, d_input.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(10.0, d_input.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(12.0, d_input.layers[0].values[1][1]);

    backpropagation_pool_layer(&layer, &d_input);

    TEST_ASSERT_EQUAL_INT(2, layer.d_input.depth);
    TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].cols_n);

    TEST_ASSERT_EQUAL_FLOAT(1.0, layer.d_input.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(2.5, layer.d_input.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(1.5, layer.d_input.layers[0].values[0][2]);

    TEST_ASSERT_EQUAL_FLOAT(3.5, layer.d_input.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0, layer.d_input.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(4.5, layer.d_input.layers[0].values[1][2]);

    TEST_ASSERT_EQUAL_FLOAT(2.5, layer.d_input.layers[0].values[2][0]);
    TEST_ASSERT_EQUAL_FLOAT(5.5, layer.d_input.layers[0].values[2][1]);
    TEST_ASSERT_EQUAL_FLOAT(3.0, layer.d_input.layers[0].values[2][2]);

    destroy_pool_layer(&layer);
    destroy_matrix3d(&d_input);
    destroy_matrix3d(&input);
    destroy_matrix3d(&output_targets);
}

void test_perceptron_or(void){
    const float learning_rate = 0.02f;
    const int iterations_n = 10000;

    matrix3d_t inputs = {0};
    matrix3d_t output_targets = {0};
    dense_layer_t input_layer = {0};
    dense_layer_t hidden_layer = {0};
    matrix2d_t d_input = {0};

    create_matrix3d(&inputs, 1, 2, 4);
    inputs.layers[0].values[0][0] = 0;
    inputs.layers[0].values[0][1] = 0;

    inputs.layers[1].values[0][0] = 0;
    inputs.layers[1].values[0][1] = 1;

    inputs.layers[2].values[0][0] = 1;
    inputs.layers[2].values[0][1] = 0;

    inputs.layers[3].values[0][0] = 1;
    inputs.layers[3].values[0][1] = 1;

    create_matrix3d(&output_targets, 1, 1, 4);
    output_targets.layers[0].values[0][0] = 0;

    output_targets.layers[1].values[0][0] = 1;

    output_targets.layers[2].values[0][0] = 1;

    output_targets.layers[3].values[0][0] = 1;
    
    init_dense_layer(&input_layer, 2, 4, ACTIVATION_TYPE_RELU);
    
    init_dense_layer(&hidden_layer, 4, 1, ACTIVATION_TYPE_RELU);

    create_matrix2d(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n);

    for(int i=0;i<iterations_n;i++){
        for(int j=0;j<inputs.depth;j++){
            feed_dense_layer(&input_layer, &inputs.layers[j]);
            process_dense_layer(&input_layer);
            feed_dense_layer(&hidden_layer, &input_layer.output_activated);
            process_dense_layer(&hidden_layer);
            compute_cost_derivative(&hidden_layer.output_activated, &output_targets.layers[j], &d_input);
            matrix2d_print(&d_input);
            backpropagation_dense_layer(&hidden_layer, &d_input, learning_rate);
            backpropagation_dense_layer(&input_layer, &hidden_layer.d_inputs, learning_rate);
        }
    }

    TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.rows_n);
    TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.cols_n);

    for(int i=0;i<inputs.depth;i++){
        feed_dense_layer(&input_layer, &inputs.layers[i]);
        process_dense_layer(&input_layer);
        feed_dense_layer(&hidden_layer, &input_layer.output_activated);
        process_dense_layer(&hidden_layer);
        printf("i: %d\tEstimated: %f\tExpected: %f\n", i, hidden_layer.output_activated.values[0][0], output_targets.layers[i].values[0][0]);
        TEST_ASSERT_FLOAT_WITHIN(0.00001, output_targets.layers[i].values[0][0], hidden_layer.output_activated.values[0][0]);
    }

    destroy_matrix3d(&inputs);
    destroy_matrix2d(&d_input);
    destroy_matrix3d(&output_targets);
    destroy_dense_layer(&input_layer);
    destroy_dense_layer(&hidden_layer);
}

void test_perceptron_and(void){
    const float learning_rate = 0.02f;
    const int iterations_n = 10000;

    matrix3d_t inputs = {0};
    matrix3d_t output_targets = {0};
    dense_layer_t input_layer = {0};
    dense_layer_t hidden_layer = {0};
    matrix2d_t d_input = {0};

    create_matrix3d(&inputs, 1, 2, 4);
    inputs.layers[0].values[0][0] = 0;
    inputs.layers[0].values[0][1] = 0;

    inputs.layers[1].values[0][0] = 0;
    inputs.layers[1].values[0][1] = 1;

    inputs.layers[2].values[0][0] = 1;
    inputs.layers[2].values[0][1] = 0;

    inputs.layers[3].values[0][0] = 1;
    inputs.layers[3].values[0][1] = 1;

    create_matrix3d(&output_targets, 1, 1, 4);
    output_targets.layers[0].values[0][0] = 0;

    output_targets.layers[1].values[0][0] = 0;

    output_targets.layers[2].values[0][0] = 0;

    output_targets.layers[3].values[0][0] = 1;
    
    init_dense_layer(&input_layer, 2, 4, ACTIVATION_TYPE_RELU);
    
    init_dense_layer(&hidden_layer, 4, 1, ACTIVATION_TYPE_RELU);

    create_matrix2d(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n);

    for(int i=0;i<iterations_n;i++){
        for(int j=0;j<inputs.depth;j++){
            feed_dense_layer(&input_layer, &inputs.layers[j]);
            process_dense_layer(&input_layer);
            feed_dense_layer(&hidden_layer, &input_layer.output_activated);
            process_dense_layer(&hidden_layer);
            compute_cost_derivative(&hidden_layer.output_activated, &output_targets.layers[j], &d_input);
            matrix2d_print(&d_input);
            backpropagation_dense_layer(&hidden_layer, &d_input, learning_rate);
            backpropagation_dense_layer(&input_layer, &hidden_layer.d_inputs, learning_rate);
        }
    }

    TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.rows_n);
    TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.cols_n);

    for(int i=0;i<inputs.depth;i++){
        feed_dense_layer(&input_layer, &inputs.layers[i]);
        process_dense_layer(&input_layer);
        feed_dense_layer(&hidden_layer, &input_layer.output_activated);
        process_dense_layer(&hidden_layer);
        printf("i: %d\tEstimated: %f\tExpected: %f\n", i, hidden_layer.output_activated.values[0][0], output_targets.layers[i].values[0][0]);
        TEST_ASSERT_FLOAT_WITHIN(0.00001, output_targets.layers[i].values[0][0], hidden_layer.output_activated.values[0][0]);
    }

    destroy_matrix3d(&inputs);
    destroy_matrix2d(&d_input);
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
    RUN_TEST(test_backpropagation_conv_layer);
    RUN_TEST(test_backpropagation_max_pool_layer);
    RUN_TEST(test_backpropagation_avg_pool_layer);
    RUN_TEST(test_perceptron_or);
    RUN_TEST(test_perceptron_and);
    int result = UNITY_END();

    return result;
}