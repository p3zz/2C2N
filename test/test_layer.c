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
    conv_layer_init(&layer, input_height, input_width, input_depth, kernel_size, kernels_n, stride, padding, ACTIVATION_TYPE_RELU);
    TEST_ASSERT_EQUAL_INT(0, layer.padding);
    TEST_ASSERT_EQUAL_INT(1, layer.stride);
    // check kernels
    TEST_ASSERT_EQUAL_INT(1, layer.kernels_n);
    TEST_ASSERT_EQUAL_INT(2, layer.kernels[0].depth);
    TEST_ASSERT_EQUAL_INT(3,  layer.kernels[0].rows_n);
    TEST_ASSERT_EQUAL_INT(3,  layer.kernels[0].cols_n);
    // check input
    TEST_ASSERT_EQUAL_INT(2, layer.input->depth);
    TEST_ASSERT_EQUAL_INT(10, layer.input->rows_n);
    TEST_ASSERT_EQUAL_INT(10, layer.input->cols_n);
    // check d_input
    TEST_ASSERT_EQUAL_INT(2, layer.d_input->depth);
    TEST_ASSERT_EQUAL_INT(10, layer.d_input->rows_n);
    TEST_ASSERT_EQUAL_INT(10, layer.d_input->cols_n);
    // check biases
    TEST_ASSERT_EQUAL_INT(8, layer.biases[0].rows_n);
    TEST_ASSERT_EQUAL_INT(8, layer.biases[0].cols_n);
    // check output
    TEST_ASSERT_EQUAL_INT(1, layer.output->depth);
    TEST_ASSERT_EQUAL_INT(8, layer.output->rows_n);
    TEST_ASSERT_EQUAL_INT(8, layer.output->cols_n);
    // check output activated
    TEST_ASSERT_EQUAL_INT(1, layer.output_activated->depth);
    TEST_ASSERT_EQUAL_INT(8, layer.output_activated->rows_n);
    TEST_ASSERT_EQUAL_INT(8, layer.output_activated->cols_n);
    conv_layer_destroy(&layer);
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
    conv_layer_init(&layer, input_height, input_width, input_depth, kernel_size, kernels_n, stride, padding, ACTIVATION_TYPE_RELU);
    float input_vals[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        9, 8, 7,
        6, 5, 4,
        3, 2, 1,
    };
    matrix3d_t input = {0};
    matrix3d_load(&input, input_height, input_width, input_depth, &input_vals[0]);

    matrix3d_set_elem(&layer.kernels[0], 0, 0, 0, 0.142);
    matrix3d_set_elem(&layer.kernels[0], 0, 1, 0, 0.607);
    matrix3d_set_elem(&layer.kernels[0], 1, 0, 0, 0.016);
    matrix3d_set_elem(&layer.kernels[0], 1, 1, 0, 0.243);
    
    matrix3d_set_elem(&layer.kernels[0], 0, 0, 1, 0.137);
    matrix3d_set_elem(&layer.kernels[0], 0, 1, 1, 0.804);
    matrix3d_set_elem(&layer.kernels[0], 1, 0, 1, 0.157);
    matrix3d_set_elem(&layer.kernels[0], 1, 1, 1, 0.401);

    matrix3d_set_elem(&layer.kernels[1], 0, 0, 0, 0.130);
    matrix3d_set_elem(&layer.kernels[1], 0, 1, 0, 0.109);
    matrix3d_set_elem(&layer.kernels[1], 1, 0, 0, 0.999);
    matrix3d_set_elem(&layer.kernels[1], 1, 1, 0, 0.218);
    
    matrix3d_set_elem(&layer.kernels[1], 0, 0, 1, 0.513);
    matrix3d_set_elem(&layer.kernels[1], 0, 1, 1, 0.839);
    matrix3d_set_elem(&layer.kernels[1], 1, 0, 1, 0.613);
    matrix3d_set_elem(&layer.kernels[1], 1, 1, 1, 0.296);
    
    matrix2d_set_elem(&layer.biases[0], 0, 0, 0.398);
    matrix2d_set_elem(&layer.biases[0], 0, 1, 0.815);
    matrix2d_set_elem(&layer.biases[0], 1, 0, 0.684);
    matrix2d_set_elem(&layer.biases[0], 1, 1, 0.911);

    matrix2d_set_elem(&layer.biases[1], 0, 0, 0.556);
    matrix2d_set_elem(&layer.biases[1], 0, 1, 0.417);
    matrix2d_set_elem(&layer.biases[1], 1, 0, 0.170);
    matrix2d_set_elem(&layer.biases[1], 1, 1, 0.613);

    conv_layer_feed(&layer, &input);
    conv_layer_forwarding(&layer);
    // printf("Kernels\n");
    // matrix3d_print(&layer.kernels[0]);
    // matrix3d_print(&layer.kernels[1]);
    // printf("Biases\n");
    // matrix2d_print(&layer.biases[0]);
    // matrix2d_print(&layer.biases[1]);
    // printf("Output\n");
    // matrix3d_print(&layer.output);
    // matrix3d_print(&layer.output_activated);

    TEST_ASSERT_EQUAL_INT(2, layer.output->depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output->rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output->cols_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output->rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output->cols_n);
    TEST_ASSERT_EQUAL_FLOAT(13.645, matrix3d_get_elem(layer.output, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(13.571, matrix3d_get_elem(layer.output, 0, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(12.458, matrix3d_get_elem(layer.output, 1, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(12.194, matrix3d_get_elem(layer.output, 1, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(22.477, matrix3d_get_elem(layer.output, 0, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(21.533, matrix3d_get_elem(layer.output, 0, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(19.676, matrix3d_get_elem(layer.output, 1, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(19.314, matrix3d_get_elem(layer.output, 1, 1, 1));

    conv_layer_destroy(&layer);
}

void test_process_pool_layer_average(void){
    pool_layer_t layer = {0};
    pool_layer_init(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_AVERAGE);
    float input_vals[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        9, 8, 7,
        6, 5, 4,
        3, 2, 1,
    };
    matrix3d_t input = {0};
    matrix3d_load(&input, 3, 3, 2, input_vals);

    pool_layer_feed(&layer, &input);
    pool_layer_forwarding(&layer);
    // printf("Input\n");
    // matrix3d_print(&input);
    // printf("Output\n");
    // matrix3d_print(&output);
    TEST_ASSERT_EQUAL_INT(2, layer.output->depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output->rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output->cols_n);
    TEST_ASSERT_EQUAL_FLOAT(3, matrix3d_get_elem(layer.output, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(4, matrix3d_get_elem(layer.output, 0, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(6, matrix3d_get_elem(layer.output, 1, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(7, matrix3d_get_elem(layer.output, 1, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(7, matrix3d_get_elem(layer.output, 0, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(6, matrix3d_get_elem(layer.output, 0, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(4, matrix3d_get_elem(layer.output, 1, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(3, matrix3d_get_elem(layer.output, 1, 1, 1));
    
    pool_layer_destroy(&layer);
}

void test_process_pool_layer_max(void){
    pool_layer_t layer = {0};
    pool_layer_init(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_MAX);
    float input_vals[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        9, 8, 7,
        6, 5, 4,
        3, 2, 1,
    };
    matrix3d_t input = {0};
    matrix3d_load(&input, 3, 3, 2, input_vals);

    pool_layer_feed(&layer, &input);
    pool_layer_forwarding(&layer);
    // printf("Input\n");
    // matrix3d_print(&input);
    // printf("Output\n");
    // matrix3d_print(&output);
    TEST_ASSERT_EQUAL_INT(2, layer.output->depth);
    TEST_ASSERT_EQUAL_INT(2, layer.output->rows_n);
    TEST_ASSERT_EQUAL_INT(2, layer.output->cols_n);
    TEST_ASSERT_EQUAL_FLOAT(5, matrix3d_get_elem(layer.output, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(6, matrix3d_get_elem(layer.output, 0, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(8, matrix3d_get_elem(layer.output, 1, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(9, matrix3d_get_elem(layer.output, 1, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(9, matrix3d_get_elem(layer.output, 0, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(8, matrix3d_get_elem(layer.output, 0, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(6, matrix3d_get_elem(layer.output, 1, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(5, matrix3d_get_elem(layer.output, 1, 1, 1));
    pool_layer_destroy(&layer);
}

// void test_process_dense_layer(void){
//     dense_layer_t layer = {0};
//     dense_layer_init(&layer, 4, 2, ACTIVATION_TYPE_RELU);

//     const float input_vals[4] = {3.f, 4.f, 2.f, 1.f};
//     matrix3d_t input = {0};
//     matrix3d_init(&input, 1, 4, 1);
//     for(int j=0;j<input.layers[0].cols_n;j++){
//         input.layers[0].values[0][j] = input_vals[j];
//     }
//     layer.weights.values[0][0] = 0.294;
//     layer.weights.values[0][1] = 0.232;
//     layer.weights.values[1][0] = 0.584;
//     layer.weights.values[1][1] = 0.244;
//     layer.weights.values[2][0] = 0.152;
//     layer.weights.values[2][1] = 0.732;
//     layer.weights.values[3][0] = 0.125;
//     layer.weights.values[3][1] = 0.793;

//     layer.biases.values[0][0] = 0.164;
//     layer.biases.values[0][1] = 0.745;

//     dense_layer_feed(&layer, &input);
    
//     dense_layer_forwarding(&layer);
//     TEST_ASSERT_EQUAL_INT(1, layer.output.layers[0].rows_n);
//     TEST_ASSERT_EQUAL_INT(2, layer.output.layers[0].cols_n);
//     TEST_ASSERT_EQUAL_FLOAT(3.811, layer.output.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(4.674, layer.output.layers[0].values[0][1]);
//     dense_layer_destroy(&layer);
//     matrix3d_destroy(&input);
// }

// void test_backpropagation_dense_layer(void){
//     dense_layer_t layer = {0};
//     dense_layer_init(&layer, 3, 2, ACTIVATION_TYPE_RELU);

//     const float input_vals[3] = {1.71f, 1.79f, 2.04f};
//     const float output_targets[2] = {1.f, 0.f};

//     matrix3d_t input = {0};
//     matrix3d_init(&input, 1, 3, 1);
//     for(int j=0;j<input.layers[0].cols_n;j++){
//         input.layers[0].values[0][j] = input_vals[j];
//     }

//     matrix3d_t output_target = {0};
//     matrix3d_init(&output_target, 1, 2, 1);
//     for(int j=0;j<output_target.layers[0].cols_n;j++){
//         output_target.layers[0].values[0][j] = output_targets[j];
//     }

//     layer.weights.values[0][0] = 0.4;
//     layer.weights.values[0][1] = 0.8;

//     layer.weights.values[1][0] = 0.6;
//     layer.weights.values[1][1] = 0.7;

//     layer.weights.values[2][0] = 0.1;
//     layer.weights.values[2][1] = 0.2;

//     layer.biases.values[0][0] = 0.3;
//     layer.biases.values[0][1] = 0.4;

//     dense_layer_feed(&layer, &input);
//     dense_layer_forwarding(&layer);

//     TEST_ASSERT_EQUAL_FLOAT(2.262f, layer.output.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(3.429f, layer.output.layers[0].values[0][1]);

//     matrix3d_t d_input = {0};
//     matrix3d_init(&d_input, output_target.layers[0].rows_n, output_target.layers[0].cols_n, 1);

//     mean_squared_error_derivative(&layer.output_activated.layers[0], &output_target.layers[0], &d_input.layers[0]);

//     TEST_ASSERT_EQUAL_FLOAT(2.524f, d_input.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(6.858f, d_input.layers[0].values[0][1]);

//     dense_layer_backpropagation(&layer, &d_input, 0.15f);

//     TEST_ASSERT_EQUAL_FLOAT(6.496, layer.d_inputs.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(6.315, layer.d_inputs.layers[0].values[0][1]);
//     TEST_ASSERT_EQUAL_FLOAT(1.624, layer.d_inputs.layers[0].values[0][2]);

//     TEST_ASSERT_EQUAL_FLOAT(-0.2474061, layer.weights.values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(-0.9590769, layer.weights.values[0][1]);
//     TEST_ASSERT_EQUAL_FLOAT(-0.077694, layer.weights.values[1][0]);
//     TEST_ASSERT_EQUAL_FLOAT(-1.141373, layer.weights.values[1][1]);

//     dense_layer_destroy(&layer);
//     matrix3d_destroy(&input);
//     matrix3d_destroy(&output_target);
//     matrix3d_destroy(&d_input);
// }

// void test_backpropagation_conv_layer(void){
//     const int input_height = 3;
//     const int input_width = 3;
//     const int input_depth = 2;
//     const int kernel_size = 2;
//     const int kernels_n = 2;
//     const int stride = 1;
//     const int padding = 0;
//     const float learning_rate = 0.05f;
//     conv_layer_t layer = {0};
//     conv_layer_init(&layer, input_height, input_width, input_depth, kernel_size, kernels_n, stride, padding, ACTIVATION_TYPE_RELU);
//     const float input_vals[2][3][3] = {
//         {
//             {1, 2, 3},
//             {4, 5, 6},
//             {7, 8, 9},
//         },
//         {
//             {9, 8, 7},
//             {6, 5, 4},
//             {3, 2, 1},
//         }
//     };
//     matrix3d_t input = {0};
//     matrix3d_init(&input, 3, 3, 2);
//     for(int i=0;i<input.depth;i++){
//         for(int j=0;j<input.layers[i].rows_n;j++){
//             for(int k=0;k<input.layers[i].cols_n;k++){
//                 input.layers[i].values[j][k] = input_vals[i][j][k];
//             }
//         }
//     }

//     layer.kernels[0].layers[0].values[0][0] = 0.985;
//     layer.kernels[0].layers[0].values[0][1] = 0.935;
//     layer.kernels[0].layers[0].values[1][0] = 0.684;
//     layer.kernels[0].layers[0].values[1][1] = 0.383;

//     layer.kernels[0].layers[1].values[0][0] = 0.750;
//     layer.kernels[0].layers[1].values[0][1] = 0.369;
//     layer.kernels[0].layers[1].values[1][0] = 0.294;
//     layer.kernels[0].layers[1].values[1][1] = 0.232;

//     layer.kernels[1].layers[0].values[0][0] = 0.584;
//     layer.kernels[1].layers[0].values[0][1] = 0.244;
//     layer.kernels[1].layers[0].values[1][0] = 0.152;
//     layer.kernels[1].layers[0].values[1][1] = 0.732;

//     layer.kernels[1].layers[1].values[0][0] = 0.125;
//     layer.kernels[1].layers[1].values[0][1] = 0.793;
//     layer.kernels[1].layers[1].values[1][0] = 0.164;
//     layer.kernels[1].layers[1].values[1][1] = 0.745;

//     layer.biases[0].values[0][0] = 0.075;
//     layer.biases[0].values[0][1] = 0.950;
//     layer.biases[0].values[1][0] = 0.053;
//     layer.biases[0].values[1][1] = 0.522;

//     layer.biases[1].values[0][0] = 0.176;
//     layer.biases[1].values[0][1] = 0.240;
//     layer.biases[1].values[1][0] = 0.798;
//     layer.biases[1].values[1][1] = 0.733;

//     conv_layer_feed(&layer, &input);
//     conv_layer_forwarding(&layer);

//     matrix3d_t output_targets = {0};
//     // the layer produces an output of 2x2x2
//     matrix3d_init(&output_targets, 2, 2, 2);
//     output_targets.layers[0].values[0][0] = 1;
//     output_targets.layers[0].values[0][1] = 1;
//     output_targets.layers[0].values[1][0] = 1;
//     output_targets.layers[0].values[1][1] = 1;
//     output_targets.layers[1].values[0][0] = 1;
//     output_targets.layers[1].values[0][1] = 1;
//     output_targets.layers[1].values[1][0] = 1;
//     output_targets.layers[1].values[1][1] = 1;
    
//     matrix3d_t d_input = {0};
//     matrix3d_init(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, layer.output_activated.depth);
//     for(int i=0;i<d_input.depth;i++){
//         mean_squared_error_derivative(&layer.output_activated.layers[i], &output_targets.layers[i], &d_input.layers[i]);
//     }
//     TEST_ASSERT_EQUAL_INT(2, d_input.depth);
//     TEST_ASSERT_EQUAL_INT(2, d_input.layers[0].rows_n);
//     TEST_ASSERT_EQUAL_INT(2, d_input.layers[0].cols_n);
//     // printf("[before] Kernel\n");
//     // matrix3d_print(&layer.kernels[0]);
//     // matrix3d_print(&layer.kernels[1]);
//     // printf("[before] bias\n");
//     // matrix2d_print(&layer.biases[0]);
//     // matrix2d_print(&layer.biases[1]);
//     conv_layer_backpropagation(&layer, &d_input, learning_rate);
//     // printf("[after] Kernel\n");
//     // matrix3d_print(&layer.kernels[0]);
//     // matrix3d_print(&layer.kernels[1]);
//     // printf("[after] bias\n");
//     // matrix2d_print(&layer.biases[0]);
//     // matrix2d_print(&layer.biases[1]);
//     // printf("[after] d_input\n");
//     // matrix3d_print(&layer.d_input);

//     matrix3d_destroy(&input);
//     matrix3d_destroy(&output_targets);
//     matrix3d_destroy(&d_input);
//     conv_layer_destroy(&layer);
//     // TEST_ASSERT_TRUE(false);
// }

// void test_backpropagation_max_pool_layer(void){
//     pool_layer_t layer = {0};
//     pool_layer_init(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_MAX);
//     const float input_vals[2][3][3] = {
//         {
//             {1, 2, 3},
//             {4, 5, 6},
//             {7, 8, 9},
//         },
//         {
//             {9, 8, 7},
//             {6, 5, 4},
//             {3, 2, 1},
//         }
//     };
//     matrix3d_t input = {0};
//     matrix3d_init(&input, 3, 3, 2);
//     for(int i=0;i<input.depth;i++){
//         for(int j=0;j<input.layers[i].rows_n;j++){
//             for(int k=0;k<input.layers[i].cols_n;k++){
//                 input.layers[i].values[j][k] = input_vals[i][j][k];
//             }
//         }
//     }

//     pool_layer_feed(&layer, &input);
//     pool_layer_forwarding(&layer);

//     matrix3d_t output_targets = {0};
//     // the layer produces an output of 2x2x2
//     matrix3d_init(&output_targets, 2, 2, 2);
//     output_targets.layers[0].values[0][0] = 1;
//     output_targets.layers[0].values[0][1] = 1;
//     output_targets.layers[0].values[1][0] = 1;
//     output_targets.layers[0].values[1][1] = 1;
//     output_targets.layers[1].values[0][0] = 1;
//     output_targets.layers[1].values[0][1] = 1;
//     output_targets.layers[1].values[1][0] = 1;
//     output_targets.layers[1].values[1][1] = 1;

//     matrix3d_t d_input = {0};
//     matrix3d_init(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, layer.output.depth);
//     for(int i=0;i<d_input.depth;i++){
//         mean_squared_error_derivative(&layer.output.layers[i], &output_targets.layers[i], &d_input.layers[i]);
//     }

//     pool_layer_backpropagation(&layer, &d_input);

//     TEST_ASSERT_EQUAL_INT(2, layer.d_input.depth);
//     TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].rows_n);
//     TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].cols_n);

//     TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[0][1]);
//     TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[0][2]);

//     TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[1][0]);
//     TEST_ASSERT_EQUAL_FLOAT(8.0, layer.d_input.layers[0].values[1][1]);
//     TEST_ASSERT_EQUAL_FLOAT(10.0, layer.d_input.layers[0].values[1][2]);

//     TEST_ASSERT_EQUAL_FLOAT(0.0, layer.d_input.layers[0].values[2][0]);
//     TEST_ASSERT_EQUAL_FLOAT(14.0, layer.d_input.layers[0].values[2][1]);
//     TEST_ASSERT_EQUAL_FLOAT(16.0, layer.d_input.layers[0].values[2][2]);

//     pool_layer_destroy(&layer);
//     matrix3d_destroy(&d_input);
//     matrix3d_destroy(&input);
//     matrix3d_destroy(&output_targets);
// }

// void test_backpropagation_avg_pool_layer(void){
//     pool_layer_t layer = {0};
//     pool_layer_init(&layer, 3, 3, 2, 2, 0, 1, POOLING_TYPE_AVERAGE);
//     const float input_vals[2][3][3] = {
//         {
//             {1, 2, 3},
//             {4, 5, 6},
//             {7, 8, 9},
//         },
//         {
//             {9, 8, 7},
//             {6, 5, 4},
//             {3, 2, 1},
//         }
//     };
//     matrix3d_t input = {0};
//     matrix3d_init(&input, 3, 3, 2);
//     for(int i=0;i<input.depth;i++){
//         for(int j=0;j<input.layers[i].rows_n;j++){
//             for(int k=0;k<input.layers[i].cols_n;k++){
//                 input.layers[i].values[j][k] = input_vals[i][j][k];
//             }
//         }
//     }

//     pool_layer_feed(&layer, &input);
//     pool_layer_forwarding(&layer);

//     matrix3d_t output_targets = {0};
//     // the layer produces an output of 2x2x2
//     matrix3d_init(&output_targets, 2, 2, 2);
//     output_targets.layers[0].values[0][0] = 1;
//     output_targets.layers[0].values[0][1] = 1;
//     output_targets.layers[0].values[1][0] = 1;
//     output_targets.layers[0].values[1][1] = 1;
//     output_targets.layers[1].values[0][0] = 1;
//     output_targets.layers[1].values[0][1] = 1;
//     output_targets.layers[1].values[1][0] = 1;
//     output_targets.layers[1].values[1][1] = 1;

//     matrix3d_t d_input = {0};
//     matrix3d_init(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, layer.output.depth);
//     for(int i=0;i<d_input.depth;i++){
//         mean_squared_error_derivative(&layer.output.layers[i], &output_targets.layers[i], &d_input.layers[i]);
//     }

//     TEST_ASSERT_EQUAL_FLOAT(4.0, d_input.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(6.0, d_input.layers[0].values[0][1]);
//     TEST_ASSERT_EQUAL_FLOAT(10.0, d_input.layers[0].values[1][0]);
//     TEST_ASSERT_EQUAL_FLOAT(12.0, d_input.layers[0].values[1][1]);

//     pool_layer_backpropagation(&layer, &d_input);

//     TEST_ASSERT_EQUAL_INT(2, layer.d_input.depth);
//     TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].rows_n);
//     TEST_ASSERT_EQUAL_INT(3, layer.d_input.layers[0].cols_n);

//     TEST_ASSERT_EQUAL_FLOAT(1.0, layer.d_input.layers[0].values[0][0]);
//     TEST_ASSERT_EQUAL_FLOAT(2.5, layer.d_input.layers[0].values[0][1]);
//     TEST_ASSERT_EQUAL_FLOAT(1.5, layer.d_input.layers[0].values[0][2]);

//     TEST_ASSERT_EQUAL_FLOAT(3.5, layer.d_input.layers[0].values[1][0]);
//     TEST_ASSERT_EQUAL_FLOAT(8.0, layer.d_input.layers[0].values[1][1]);
//     TEST_ASSERT_EQUAL_FLOAT(4.5, layer.d_input.layers[0].values[1][2]);

//     TEST_ASSERT_EQUAL_FLOAT(2.5, layer.d_input.layers[0].values[2][0]);
//     TEST_ASSERT_EQUAL_FLOAT(5.5, layer.d_input.layers[0].values[2][1]);
//     TEST_ASSERT_EQUAL_FLOAT(3.0, layer.d_input.layers[0].values[2][2]);

//     pool_layer_destroy(&layer);
//     matrix3d_destroy(&d_input);
//     matrix3d_destroy(&input);
//     matrix3d_destroy(&output_targets);
// }

// void test_perceptron_or(void){
//     const float learning_rate = 0.02f;
//     const int iterations_n = 10000;

//     int inputs_n = 4;
//     matrix3d_t* inputs = (matrix3d_t*)malloc(inputs_n * sizeof(matrix3d_t));
//     matrix3d_t output_targets = {0};
//     dense_layer_t input_layer = {0};
//     dense_layer_t hidden_layer = {0};
//     matrix3d_t d_input = {0};

//     for(int i=0;i<inputs_n;i++){
//         matrix3d_init(&inputs[i], 1, 2, 1);
//     }

//     inputs[0].layers[0].values[0][0] = 0;
//     inputs[0].layers[0].values[0][1] = 0;

//     inputs[1].layers[0].values[0][0] = 0;
//     inputs[1].layers[0].values[0][1] = 1;

//     inputs[2].layers[0].values[0][0] = 1;
//     inputs[2].layers[0].values[0][1] = 0;

//     inputs[3].layers[0].values[0][0] = 1;
//     inputs[3].layers[0].values[0][1] = 1;

//     matrix3d_init(&output_targets, 1, 1, 4);
//     output_targets.layers[0].values[0][0] = 0;

//     output_targets.layers[1].values[0][0] = 1;

//     output_targets.layers[2].values[0][0] = 1;

//     output_targets.layers[3].values[0][0] = 1;
    
//     dense_layer_init(&input_layer, 2, 4, ACTIVATION_TYPE_RELU);
    
//     dense_layer_init(&hidden_layer, 4, 1, ACTIVATION_TYPE_RELU);

//     matrix3d_init(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, 1);

//     for(int i=0;i<iterations_n;i++){
//         for(int j=0;j<inputs_n;j++){
//             dense_layer_feed(&input_layer, &inputs[j]);
//             dense_layer_forwarding(&input_layer);
//             dense_layer_feed(&hidden_layer, &input_layer.output_activated);
//             dense_layer_forwarding(&hidden_layer);
//             mean_squared_error_derivative(&hidden_layer.output_activated.layers[0], &output_targets.layers[j], &d_input.layers[0]);
//             dense_layer_backpropagation(&hidden_layer, &d_input, learning_rate);
//             dense_layer_backpropagation(&input_layer, &hidden_layer.d_inputs, learning_rate);
//         }
//     }

//     TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.layers[0].rows_n);
//     TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.layers[0].cols_n);

//     for(int i=0;i<inputs_n;i++){
//         dense_layer_feed(&input_layer, &inputs[i]);
//         dense_layer_forwarding(&input_layer);
//         dense_layer_feed(&hidden_layer, &input_layer.output_activated);
//         dense_layer_forwarding(&hidden_layer);
//         printf("Target: %f\tprediction: %f\n", output_targets.layers[i].values[0][0], hidden_layer.output_activated.layers[0].values[0][0]);
//         // TEST_ASSERT_FLOAT_WITHIN(0.00001, output_targets.layers[i].values[0][0], hidden_layer.output_activated.layers[0].values[0][0]);
//     }

//     for(int i=0;i<inputs_n;i++){
//         matrix3d_destroy(&inputs[i]);        
//     }
//     free(inputs);
//     matrix3d_destroy(&d_input);
//     matrix3d_destroy(&output_targets);
//     dense_layer_destroy(&input_layer);
//     dense_layer_destroy(&hidden_layer);

//     // TEST_ASSERT_TRUE(false);
// }

// void test_perceptron_and(void){
//     const float learning_rate = 0.02f;
//     const int iterations_n = 10000;

//     int inputs_n = 4;
//     matrix3d_t* inputs = (matrix3d_t*)malloc(inputs_n * sizeof(matrix3d_t));
//     matrix3d_t output_targets = {0};
//     dense_layer_t input_layer = {0};
//     dense_layer_t hidden_layer = {0};
//     matrix3d_t d_input = {0};

//     for(int i=0;i<inputs_n;i++){
//         matrix3d_init(&inputs[i], 1, 2, 1);
//     }

//     inputs[0].layers[0].values[0][0] = 0;
//     inputs[0].layers[0].values[0][1] = 0;

//     inputs[1].layers[0].values[0][0] = 0;
//     inputs[1].layers[0].values[0][1] = 1;

//     inputs[2].layers[0].values[0][0] = 1;
//     inputs[2].layers[0].values[0][1] = 0;

//     inputs[3].layers[0].values[0][0] = 1;
//     inputs[3].layers[0].values[0][1] = 1;

//     matrix3d_init(&output_targets, 1, 1, 4);
//     output_targets.layers[0].values[0][0] = 0;

//     output_targets.layers[1].values[0][0] = 0;

//     output_targets.layers[2].values[0][0] = 0;

//     output_targets.layers[3].values[0][0] = 1;
    
//     dense_layer_init(&input_layer, 2, 4, ACTIVATION_TYPE_RELU);
    
//     dense_layer_init(&hidden_layer, 4, 1, ACTIVATION_TYPE_RELU);

//     matrix3d_init(&d_input, output_targets.layers[0].rows_n, output_targets.layers[0].cols_n, 1);

//     for(int i=0;i<iterations_n;i++){
//         for(int j=0;j<inputs_n;j++){
//             dense_layer_feed(&input_layer, &inputs[j]);
//             dense_layer_forwarding(&input_layer);
//             dense_layer_feed(&hidden_layer, &input_layer.output_activated);
//             dense_layer_forwarding(&hidden_layer);
//             mean_squared_error_derivative(&hidden_layer.output_activated.layers[0], &output_targets.layers[j], &d_input.layers[0]);
//             dense_layer_backpropagation(&hidden_layer, &d_input, learning_rate);
//             dense_layer_backpropagation(&input_layer, &hidden_layer.d_inputs, learning_rate);
//         }
//     }

//     TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.layers[0].rows_n);
//     TEST_ASSERT_EQUAL_INT(1, hidden_layer.output_activated.layers[0].cols_n);

//     for(int i=0;i<inputs_n;i++){
//         dense_layer_feed(&input_layer, &inputs[i]);
//         dense_layer_forwarding(&input_layer);
//         dense_layer_feed(&hidden_layer, &input_layer.output_activated);
//         dense_layer_forwarding(&hidden_layer);
//         TEST_ASSERT_FLOAT_WITHIN(0.00001, output_targets.layers[i].values[0][0], hidden_layer.output_activated.layers[0].values[0][0]);
//     }

//     for(int i=0;i<inputs_n;i++){
//         matrix3d_destroy(&inputs[i]);        
//     }
//     free(inputs);
//     matrix3d_destroy(&d_input);
//     matrix3d_destroy(&output_targets);
//     dense_layer_destroy(&input_layer);
//     dense_layer_destroy(&hidden_layer);
// }

// // https://medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b
// // LeNet-5 CNN
// // layer 0: convolutional layer / input (32x32x1) output(28x28x6) kernel(5, 5, 6) padding 0 stride 1
// // layer 1: average pooling layer / input (28x28x6) output(14x14x6) kernel(2, 2, 6) padding 0 stride 2
// // layer 2: convolutional layer / input (14x14x6) output(10x10x16) kernel(5, 5, 16) padding 0 stride 1
// // layer 3: average pooling layer / input (10x10x16) output(5x5x16) kernel(2, 2, 16) padding 0 stride 2
// // reshape: input (5x5x16) output(400x1x1)
// // layer 4: fully connected layer / input (400) output(120)
// // layer 5: fully connected layer / input (120) output(84)
// // layer 6: fully connected layer / input (84) output(10)
// void test_lenet5_cnn(void){
//     const float learning_rate = 0.02;
//     const int iterations = 10;
//     conv_layer_t layer0 = {0};
//     pool_layer_t layer1 = {0};
//     conv_layer_t layer2 = {0};
//     pool_layer_t layer3 = {0};
//     dense_layer_t layer4 = {0};
//     dense_layer_t layer5 = {0};
//     dense_layer_t layer6 = {0};
//     softmax_layer_t layer7 = {0};
    
//     matrix3d_t input = {0};
//     matrix3d_t d_input = {0};
//     matrix3d_t output_target = {0};
//     matrix3d_init(&input, 32, 32, 1);
//     matrix3d_init(&output_target, 1, 10, 1);
//     matrix3d_init(&d_input, 1, 10, 1);

//     conv_layer_init(&layer0, 32, 32, 1, 5, 6, 1, 0, ACTIVATION_TYPE_TANH);
//     pool_layer_init(&layer1, 28, 28, 6, 2, 0, 2, POOLING_TYPE_AVERAGE);
//     conv_layer_init(&layer2, 14, 14, 6, 5, 16, 1, 0, ACTIVATION_TYPE_TANH);
//     pool_layer_init(&layer3, 10, 10, 16, 2, 0, 2, POOLING_TYPE_AVERAGE);
//     dense_layer_init(&layer4, 400, 120, ACTIVATION_TYPE_TANH);
//     dense_layer_init(&layer5, 120, 84, ACTIVATION_TYPE_TANH);
//     dense_layer_init(&layer6, 84, 10, ACTIVATION_TYPE_IDENTITY);
//     softmax_layer_init(&layer7, 10);

//     // draw the digit 1
//     for(int i=0;i<input.layers[0].rows_n;i++){
//         for(int j=0;j<input.layers[0].cols_n;j++){
//             if(j == input.layers[0].cols_n / 2){
//                 input.layers[0].values[i][j] = 255;
//             }
//         }
//     }

//     // printf("Height: %d\tWidth: %d\tDepth: %d\n", layer1.output.layers[0].rows_n, layer1.output.layers[0].cols_n, layer1.output.depth);

//     // printf("Input");
//     // matrix3d_print(&input);

//     matrix3d_t aux = {0};
//     matrix3d_init(&aux, 5, 5, 16);

//     for(int i=0;i<iterations;i++){
//         conv_layer_feed(&layer0, &input);
//         conv_layer_forwarding(&layer0);
//         // printf("Conv layer 0-----------------------------\n");
//         // matrix3d_print(&layer0.output);
//         // matrix3d_print(&layer0.output_activated);

//         pool_layer_feed(&layer1, &layer0.output_activated);
//         pool_layer_forwarding(&layer1);
//         // printf("Pool layer 1-----------------------------\n");
//         // matrix3d_print(&layer1.output);

//         conv_layer_feed(&layer2, &layer1.output);
//         conv_layer_forwarding(&layer2);
//         // printf("Conv layer 2-----------------------------\n");
//         // matrix3d_print(&layer2.output);
//         // matrix3d_print(&layer2.output_activated);

//         pool_layer_feed(&layer3, &layer2.output_activated);
//         pool_layer_forwarding(&layer3);
//         // printf("Pool layer 3-----------------------------\n");
//         // matrix3d_print(&layer3.output);

//         matrix3d_reshape(&layer3.output, &layer4.inputs);
//         dense_layer_forwarding(&layer4);
//         // printf("Dense layer 4-----------------------------\n");
//         // matrix3d_print(&layer4.output);
//         // matrix3d_print(&layer4.output_activated);
        
//         dense_layer_feed(&layer5, &layer4.output_activated);
//         dense_layer_forwarding(&layer5);
//         // printf("Dense layer 5-----------------------------\n");
//         // matrix3d_print(&layer5.output);
//         // matrix3d_print(&layer5.output_activated);

//         dense_layer_feed(&layer6, &layer5.output_activated);
//         dense_layer_forwarding(&layer6);
//         // printf("Dense layer 6-----------------------------\n");
//         // matrix3d_print(&layer6.output);
//         // matrix3d_print(&layer6.output_activated);

//         // printf("Softmax layer 7-----------------------------\n");
//         softmax_layer_feed(&layer7, &layer6.output_activated);
//         softmax_layer_forwarding(&layer7);
//         // matrix3d_print(&layer7.output);

//         output_target.layers[0].values[0][1] = 1.0;
//         mean_squared_error_derivative(&layer7.output.layers[0], &output_target.layers[0], &d_input.layers[0]);
//         // matrix3d_print(&d_input);

//         // printf("[BACKPROP] Softmax layer 7-----------------------------\n");
//         softmax_layer_backpropagation(&layer7, &d_input);
//         // matrix3d_print(&layer7.d_input);

//         // printf("[BACKPROP] Dense layer 6-----------------------------\n");
//         dense_layer_backpropagation(&layer6, &layer7.d_input, learning_rate);
//         // matrix3d_print(&layer6.d_inputs);

//         // printf("[BACKPROP] Dense layer 5-----------------------------\n");
//         dense_layer_backpropagation(&layer5, &layer6.d_inputs, learning_rate);
//         // matrix3d_print(&layer5.d_inputs);
        
//         // printf("[BACKPROP] Dense layer 4-----------------------------\n");
//         dense_layer_backpropagation(&layer4, &layer5.d_inputs, learning_rate);
//         // matrix3d_print(&layer4.d_inputs);

//         matrix3d_reshape(&layer4.d_inputs, &aux);
        
//         pool_layer_backpropagation(&layer3, &aux);
//         // matrix3d_print(&layer3.d_input);
        
//         conv_layer_backpropagation(&layer2, &layer3.d_input, learning_rate);
//         // matrix3d_print(&layer2.d_input);
//         pool_layer_backpropagation(&layer1, &layer2.d_input);
//         // matrix3d_print(&layer1.d_input);
//         conv_layer_backpropagation(&layer0, &layer1.d_input, learning_rate);
//     }

//     matrix2d_print(&d_input.layers[0]);

//     matrix3d_destroy(&aux);

//     conv_layer_destroy(&layer0);
//     pool_layer_destroy(&layer1);
//     conv_layer_destroy(&layer2);
//     pool_layer_destroy(&layer3);
//     dense_layer_destroy(&layer4);
//     dense_layer_destroy(&layer5);
//     dense_layer_destroy(&layer6);
//     softmax_layer_destroy(&layer7);
//     matrix3d_destroy(&input);
//     matrix3d_destroy(&d_input);
//     matrix3d_destroy(&output_target);

//     // TEST_ASSERT_TRUE(false);
// }

int main(void)
{
    srand(0);
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_init_conv_layer);
    RUN_TEST(test_process_conv_layer);
    RUN_TEST(test_process_pool_layer_average);
    RUN_TEST(test_process_pool_layer_max);
    // RUN_TEST(test_process_dense_layer);
    // RUN_TEST(test_backpropagation_dense_layer);
    // RUN_TEST(test_backpropagation_conv_layer);
    // RUN_TEST(test_backpropagation_max_pool_layer);
    // RUN_TEST(test_backpropagation_avg_pool_layer);
    // RUN_TEST(test_perceptron_or);
    // RUN_TEST(test_perceptron_and);
    // RUN_TEST(test_lenet5_cnn);
    int result = UNITY_END();


    return result;
}