#include <unity.h>
#include <stdbool.h>
#include "common.h"
#include "string.h"
#include "stdlib.h"
#include "utils.h"

void setUp()
{

}

void tearDown()
{

}

float* get_ref(float* arr, int i){
    return &arr[i];
}

void test_always_true(void){
    TEST_ASSERT_TRUE(true);
}

void test_matrix2d_get_elem_as_ref(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    const float* ref = matrix2d_get_elem_as_ref(&m1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.5f, *ref);
}

void test_matrix2d_get_elem_as_mut_ref(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    float* ref = matrix2d_get_elem_as_mut_ref(&m1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.5f, *ref);
}

void test_matrix2d_get_elem(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    float ref = matrix2d_get_elem(&m1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.5f, ref);
}

void test_matrix2d_set_elem(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    matrix2d_set_elem(&m1, 0, 2, 3.5f);
    float res = matrix2d_get_elem(&m1, 0, 2);
    TEST_ASSERT_EQUAL_FLOAT(3.5f, res);
}

void test_matrix3d_get_elem_as_ref(void){
    matrix3d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    const float* ref = matrix3d_get_elem_as_ref(&m1, 0, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(4.f, *ref);
}

void test_matrix3d_get_elem_as_mut_ref(void){
    matrix3d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    float* ref = matrix3d_get_elem_as_mut_ref(&m1, 0, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(4.f, *ref);
}

void test_matrix3d_get_elem(void){
    matrix3d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    float ref = matrix3d_get_elem(&m1, 0, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(4.f, ref);
}

void test_matrix3d_get_slice_as_mut_ref(void){
    matrix3d_t m1 = {0};
    matrix2d_t slice = {0};
    
    float content[] = {
        3.f, 1.f, 2.f,
        5.f, 6.f, 7.f,

        9.f, 4.f, 5.f,
        12.f, 11.f, 10.f
    };
    matrix3d_load(&m1, 2, 3, 2, content);
    matrix3d_get_slice_as_mut_ref(&m1, &slice, 1);
    float ref = matrix2d_get_elem(&slice, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(11.f, ref);
}

void test_common_cross_correlation_nopadding(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {0};

    float m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);
    
    float m2_values[] = {
        4, 3,
        3, 2
    };
    matrix2d_load(&m2, 2, 2, &m2_values[0]);

    int padding = 0;
    int stride = 1;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(43.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(74.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(43.f, matrix2d_get_elem(&result, 1, 1));

    matrix2d_destroy(&result);
}

void test_common_cross_correlation_padding(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    float m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);

    float m2_values[] = {
        4, 3,
        3, 2
    };
    matrix2d_load(&m2, 2, 2, &m2_values[0]);

    int padding = 1;
    int stride = 1;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(4, result.rows_n);
    TEST_ASSERT_EQUAL_INT(4, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(8.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(18.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(25.f, matrix2d_get_elem(&result, 0, 2));
    TEST_ASSERT_EQUAL_FLOAT(24.f, matrix2d_get_elem(&result, 0, 3));

    TEST_ASSERT_EQUAL_FLOAT(30.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(54.f, matrix2d_get_elem(&result, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(43.f, matrix2d_get_elem(&result, 1, 2));
    TEST_ASSERT_EQUAL_FLOAT(38.f, matrix2d_get_elem(&result, 1, 3));

    TEST_ASSERT_EQUAL_FLOAT(41.f, matrix2d_get_elem(&result, 2, 0));
    TEST_ASSERT_EQUAL_FLOAT(74.f, matrix2d_get_elem(&result, 2, 1));
    TEST_ASSERT_EQUAL_FLOAT(43.f, matrix2d_get_elem(&result, 2, 2));
    TEST_ASSERT_EQUAL_FLOAT(26.f, matrix2d_get_elem(&result, 2, 3));

    TEST_ASSERT_EQUAL_FLOAT(21.f, matrix2d_get_elem(&result, 3, 0));
    TEST_ASSERT_EQUAL_FLOAT(49.f, matrix2d_get_elem(&result, 3, 1));
    TEST_ASSERT_EQUAL_FLOAT(46.f, matrix2d_get_elem(&result, 3, 2));
    TEST_ASSERT_EQUAL_FLOAT(24.f, matrix2d_get_elem(&result, 3, 3));
    
    matrix2d_destroy(&result);
}

void test_common_cross_correlation_nopadding_stride(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};
    
    float m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);
    
    float m2_values[] = {
        4, 3,
        3, 2
    };
    matrix2d_load(&m2, 2, 2, &m2_values[0]);
    
    int padding = 0;
    int stride = 2;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(1, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, matrix2d_get_elem(&result, 0, 0));

    matrix2d_destroy(&result);
}

void test_common_cross_correlation_padding_stride(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    float m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);
    
    float m2_values[] = {
        4, 3,
        3, 2
    };
    matrix2d_load(&m2, 2, 2, &m2_values[0]);

    int padding = 1;
    int stride = 2;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(8.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(25.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(41.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(43.f, matrix2d_get_elem(&result, 1, 1));

    matrix2d_destroy(&result);
}

void test_common_max_pooling(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};
    matrix3d_t indexes = {0};

    float m_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m, 3, 3, &m_values[0]);

    int kernel_size = 2;
    int padding = 0;
    int stride = 1;

    int result_height;
    int result_width;
    compute_output_size(m.rows_n, m.cols_n, kernel_size, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);

    matrix3d_init(&indexes, result_height, result_width, 2);

    max_pooling(&m, &result, &indexes, kernel_size, padding, stride);

    TEST_ASSERT_EQUAL_INT(2, indexes.rows_n);
    TEST_ASSERT_EQUAL_INT(2, indexes.cols_n);

    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);

    TEST_ASSERT_EQUAL_FLOAT(9.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_INT(1, matrix3d_get_elem(&indexes, 0, 0, 0));
    TEST_ASSERT_EQUAL_INT(0, matrix3d_get_elem(&indexes, 0, 0, 1));

    TEST_ASSERT_EQUAL_FLOAT(8.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_INT(0, matrix3d_get_elem(&indexes, 0, 1, 0));
    TEST_ASSERT_EQUAL_INT(2, matrix3d_get_elem(&indexes, 0, 1, 1));

    TEST_ASSERT_EQUAL_FLOAT(9.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_INT(1, matrix3d_get_elem(&indexes, 1, 0, 0));
    TEST_ASSERT_EQUAL_INT(0, matrix3d_get_elem(&indexes, 1, 0, 1));

    TEST_ASSERT_EQUAL_FLOAT(7.f, matrix2d_get_elem(&result, 1, 1));
    TEST_ASSERT_EQUAL_INT(2, matrix3d_get_elem(&indexes, 1, 1, 0));
    TEST_ASSERT_EQUAL_INT(1, matrix3d_get_elem(&indexes, 1, 1, 1));

    matrix2d_destroy(&result);
    matrix3d_destroy(&indexes);
}

void test_common_avg_pooling(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    float m_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m, 3, 3, &m_values[0]);

    int kernel_size = 2;
    int padding = 0;
    int stride = 1;

    int result_height;
    int result_width;
    compute_output_size(m.rows_n, m.cols_n, kernel_size, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);

    avg_pooling(&m, &result, kernel_size, padding, stride);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.25f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(3.5f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(6.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix2d_get_elem(&result, 1, 1));

    matrix2d_destroy(&result);
}

void test_common_matrix2d_reshape(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    float m_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
    };
    matrix2d_load(&m, 2, 3, &m_values[0]);

    matrix2d_reshape(&m, &result, 3, 2);
    TEST_ASSERT_EQUAL_INT(3, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(3.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(8.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(9.f, matrix2d_get_elem(&result, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(1.f, matrix2d_get_elem(&result, 2, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix2d_get_elem(&result, 2, 1));

    matrix2d_destroy(&result);
}

void test_common_matrix2d_reshape_2(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    float m_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m, 3, 3, &m_values[0]);

    matrix2d_reshape(&m, &result, 1, 9);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(9, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(1.f, matrix2d_get_elem(&result, 0, 4));
    TEST_ASSERT_EQUAL_FLOAT(7.f, matrix2d_get_elem(&result, 0, 6));
    TEST_ASSERT_EQUAL_FLOAT(6.f, matrix2d_get_elem(&result, 0, 8));

    matrix2d_destroy(&result);
}

void test_common_matrix3d_reshape(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    float m_values[] = {
        4.f, 3.f,
        9.f, 1.f,
        5.f, 2.f,

        9.f, 2.f,
        7.f, 1.f,
        3.f, 2.f,
    };
    matrix3d_load(&m, 3, 2, 2, &m_values[0]);

    matrix3d_init(&result, 1, 12, 1);
    matrix3d_reshape(&m, &result);

    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix3d_get_elem(&result, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(1.f, matrix3d_get_elem(&result, 0, 3, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 0, 7, 0));
    TEST_ASSERT_EQUAL_FLOAT(7.f, matrix3d_get_elem(&result, 0, 8, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 0, 11, 0));

    matrix3d_destroy(&result);
}

void test_common_matrix3d_reshape_2(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    float m_values[] = {
        4.f, 3.f, 9.f, 1.f, 5.f, 2.f, 9.f, 2.f, 7.f, 1.f, 3.f, 2.f
    };

    matrix3d_load(&m, 1, 12, 1, &m_values[0]);
    matrix3d_init(&result, 3, 2, 2);

    matrix3d_reshape(&m, &result);
    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix3d_get_elem(&result, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(3.f, matrix3d_get_elem(&result, 0, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(9.f, matrix3d_get_elem(&result, 1, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 2, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 0, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(3.f, matrix3d_get_elem(&result, 2, 0, 1));

    matrix3d_destroy(&result);
}

void test_common_matrix2d_softmax_inplace(void){
    matrix2d_t m = {0};
    float m_values[] = {2.f, 1.f, 0.1f};
    
    matrix2d_load(&m, 1, 3, &m_values[0]);

    matrix2d_softmax_inplace(&m);
    TEST_ASSERT_EQUAL_FLOAT(0.5016878, matrix2d_get_elem(&m, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(0.304289, matrix2d_get_elem(&m, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.1940232, matrix2d_get_elem(&m, 0, 2));
}

void test_parse_line(void){
    char str[] = "1,255,12,123,2";
    int length = sizeof(str)/sizeof(char);
    
    matrix2d_t res = {0};
    float label = 0.f;

    matrix2d_init(&res, 2, 2);
    parse_line(str, length, &res, &label);
    TEST_ASSERT_EQUAL_FLOAT(1.f, label);
    TEST_ASSERT_EQUAL_FLOAT(255.f, matrix2d_get_elem(&res, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(12.f, matrix2d_get_elem(&res, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(123.f, matrix2d_get_elem(&res, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix2d_get_elem(&res, 1, 1));

    matrix2d_destroy(&res);
}

void test_zero_pad(void){
    matrix2d_t m = {0};
    matrix2d_t res = {0};

    float m_values[] = {
        1.f, 0.f, 3.f,
        0.f, 4.f, 4.f,
        9.f, 0.f, 0.f
    };

    matrix2d_load(&m, 3, 3, &m_values[0]);
    matrix2d_init(&res, 7, 7);

    zero_pad(&m, &res, 2);
    
    matrix2d_destroy(&res);
}

void test_matrix2d_load(void){
    const int height = 2;
    const int width = 2;
    matrix2d_t m = {0};
    float data[] = {1.f, 2.f, 3.f, 4.f};
    float* ptr = &data[0];

    matrix2d_load(&m, height, width, ptr);
    TEST_ASSERT_EQUAL_FLOAT(data[0], matrix2d_get_elem(&m, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(data[1], matrix2d_get_elem(&m, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(data[2], matrix2d_get_elem(&m, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(data[3], matrix2d_get_elem(&m, 1, 1));
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_common_cross_correlation_nopadding);
    RUN_TEST(test_common_cross_correlation_padding);
    RUN_TEST(test_common_cross_correlation_nopadding_stride);
    RUN_TEST(test_common_cross_correlation_padding_stride);
    RUN_TEST(test_common_max_pooling);
    RUN_TEST(test_common_avg_pooling);
    RUN_TEST(test_common_matrix2d_reshape);
    RUN_TEST(test_common_matrix2d_reshape_2);
    RUN_TEST(test_common_matrix3d_reshape);
    RUN_TEST(test_common_matrix3d_reshape_2);
    RUN_TEST(test_common_matrix2d_softmax_inplace);
    RUN_TEST(test_parse_line);
    RUN_TEST(test_zero_pad);
    RUN_TEST(test_matrix2d_load);
    RUN_TEST(test_matrix2d_get_elem_as_ref);
    RUN_TEST(test_matrix2d_get_elem_as_mut_ref);
    RUN_TEST(test_matrix2d_get_elem);
    RUN_TEST(test_matrix2d_set_elem);
    RUN_TEST(test_matrix3d_get_elem_as_ref);
    RUN_TEST(test_matrix3d_get_elem_as_mut_ref);
    RUN_TEST(test_matrix3d_get_elem);
    RUN_TEST(test_matrix3d_get_slice_as_mut_ref);
    
    int result = UNITY_END();

    return result;
}