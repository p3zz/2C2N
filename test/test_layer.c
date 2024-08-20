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

int main(void)
{
    srand(0);
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_init_conv_layer);
    int result = UNITY_END();

    return result;
}