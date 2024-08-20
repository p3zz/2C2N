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
    init_conv_layer(&layer, 3, 2, 1, 1, 0);
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

void test_feed_forward(void){
    conv_layer_t layer = {0};
    init_conv_layer(&layer, 2, 2, 2, 1, 0);
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

    matrix3d_t output = {0};

    feed_forward(&layer, &input, &output);
    printf("Input\n");
    matrix3d_print(&input);
    printf("Kernels\n");
    matrix3d_print(&layer.kernels[0]);
    matrix3d_print(&layer.kernels[1]);
    printf("Output\n");
    matrix3d_print(&output);
    TEST_ASSERT_EQUAL_INT(2, output.depth);
    TEST_ASSERT_EQUAL_INT(2, output.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, output.layers[0].cols_n);
    TEST_ASSERT_EQUAL_INT(2, output.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(2, output.layers[1].cols_n);
    TEST_ASSERT(false);
}

int main(void)
{
    srand(0);
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_init_conv_layer);
    RUN_TEST(test_feed_forward);
    int result = UNITY_END();

    return result;
}