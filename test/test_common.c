#include <unity.h>
#include <stdbool.h>
#include "env.h"
#include "common.h"
#include "string.h"
#include "stdlib.h"
#include "utils.h"

#if EMBEDDED_ENV
#define TEST_ASSERT_MATRIX_VALUE(expected, value) TEST_ASSERT_EQUAL_UINT32(expected, value)
#else
#define TEST_ASSERT_MATRIX_VALUE(expected, value) TEST_ASSERT_EQUAL_FLOAT(expected, value)
#endif

void setUp()
{

}

void tearDown()
{

}

void test_always_true(void){
    TEST_ASSERT_TRUE(true);
}

#if !EMBEDDED_ENV

void test_matrix2d_get_elem_as_ref(void){
    matrix2d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    const matrix_type* ref = matrix2d_get_elem_as_ref(&m1, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(13.5f, *ref);
}

void test_matrix2d_get_elem_as_mut_ref(void){
    matrix2d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    matrix_type* ref = matrix2d_get_elem_as_mut_ref(&m1, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(13.5f, *ref);
}

void test_matrix2d_get_elem(void){
    matrix2d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    matrix_type ref = matrix2d_get_elem(&m1, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(13.5f, ref);
}

void test_matrix2d_set_elem(void){
    matrix2d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    matrix2d_set_elem(&m1, 0, 2, 3.5f);
    matrix_type res = matrix2d_get_elem(&m1, 0, 2);
    TEST_ASSERT_MATRIX_VALUE(3.5f, res);
}

void test_matrix3d_get_elem_as_ref(void){
    matrix3d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    const matrix_type* ref = matrix3d_get_elem_as_ref(&m1, 0, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(4.f, *ref);
}

void test_matrix3d_get_elem_as_mut_ref(void){
    matrix3d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    matrix_type* ref = matrix3d_get_elem_as_mut_ref(&m1, 0, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(4.f, *ref);
}

void test_matrix3d_get_elem(void){
    matrix3d_t m1 = {0};
    matrix_type content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    matrix_type ref = matrix3d_get_elem(&m1, 0, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(4.f, ref);
}

void test_matrix3d_get_slice_as_mut_ref(void){
    matrix3d_t m1 = {0};
    matrix2d_t slice = {0};
    
    matrix_type content[] = {
        3.f, 1.f, 2.f,
        5.f, 6.f, 7.f,

        9.f, 4.f, 5.f,
        12.f, 11.f, 10.f
    };
    matrix3d_load(&m1, 2, 3, 2, content);
    matrix3d_get_slice_as_mut_ref(&m1, &slice, 1);
    matrix_type ref = matrix2d_get_elem(&slice, 1, 1);
    TEST_ASSERT_MATRIX_VALUE(11.f, ref);
}

void test_common_cross_correlation_nopadding(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {0};

    matrix_type m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);
    
    matrix_type m2_values[] = {
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
    TEST_ASSERT_MATRIX_VALUE(54.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(43.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(74.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(43.f, matrix2d_get_elem(&result, 1, 1));

    matrix2d_destroy(&result);
}

void test_common_cross_correlation_padding(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    matrix_type m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);

    matrix_type m2_values[] = {
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
    TEST_ASSERT_MATRIX_VALUE(8.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(18.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(25.f, matrix2d_get_elem(&result, 0, 2));
    TEST_ASSERT_MATRIX_VALUE(24.f, matrix2d_get_elem(&result, 0, 3));

    TEST_ASSERT_MATRIX_VALUE(30.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(54.f, matrix2d_get_elem(&result, 1, 1));
    TEST_ASSERT_MATRIX_VALUE(43.f, matrix2d_get_elem(&result, 1, 2));
    TEST_ASSERT_MATRIX_VALUE(38.f, matrix2d_get_elem(&result, 1, 3));

    TEST_ASSERT_MATRIX_VALUE(41.f, matrix2d_get_elem(&result, 2, 0));
    TEST_ASSERT_MATRIX_VALUE(74.f, matrix2d_get_elem(&result, 2, 1));
    TEST_ASSERT_MATRIX_VALUE(43.f, matrix2d_get_elem(&result, 2, 2));
    TEST_ASSERT_MATRIX_VALUE(26.f, matrix2d_get_elem(&result, 2, 3));

    TEST_ASSERT_MATRIX_VALUE(21.f, matrix2d_get_elem(&result, 3, 0));
    TEST_ASSERT_MATRIX_VALUE(49.f, matrix2d_get_elem(&result, 3, 1));
    TEST_ASSERT_MATRIX_VALUE(46.f, matrix2d_get_elem(&result, 3, 2));
    TEST_ASSERT_MATRIX_VALUE(24.f, matrix2d_get_elem(&result, 3, 3));
    
    matrix2d_destroy(&result);
}

void test_common_cross_correlation_nopadding_stride(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};
    
    matrix_type m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);
    
    matrix_type m2_values[] = {
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
    TEST_ASSERT_MATRIX_VALUE(54.f, matrix2d_get_elem(&result, 0, 0));

    matrix2d_destroy(&result);
}

void test_common_cross_correlation_padding_stride(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    matrix_type m1_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m1, 3, 3, &m1_values[0]);
    
    matrix_type m2_values[] = {
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
    TEST_ASSERT_MATRIX_VALUE(8.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(25.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(41.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(43.f, matrix2d_get_elem(&result, 1, 1));

    matrix2d_destroy(&result);
}

void test_common_max_pooling(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};
    matrix3d_t indexes = {0};

    matrix_type m_values[] = {
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

    TEST_ASSERT_MATRIX_VALUE(9.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_EQUAL_INT(1, matrix3d_get_elem(&indexes, 0, 0, 0));
    TEST_ASSERT_EQUAL_INT(0, matrix3d_get_elem(&indexes, 0, 0, 1));

    TEST_ASSERT_MATRIX_VALUE(8.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_EQUAL_INT(0, matrix3d_get_elem(&indexes, 0, 1, 0));
    TEST_ASSERT_EQUAL_INT(2, matrix3d_get_elem(&indexes, 0, 1, 1));

    TEST_ASSERT_MATRIX_VALUE(9.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_EQUAL_INT(1, matrix3d_get_elem(&indexes, 1, 0, 0));
    TEST_ASSERT_EQUAL_INT(0, matrix3d_get_elem(&indexes, 1, 0, 1));

    TEST_ASSERT_MATRIX_VALUE(7.f, matrix2d_get_elem(&result, 1, 1));
    TEST_ASSERT_EQUAL_INT(2, matrix3d_get_elem(&indexes, 1, 1, 0));
    TEST_ASSERT_EQUAL_INT(1, matrix3d_get_elem(&indexes, 1, 1, 1));

    matrix2d_destroy(&result);
    matrix3d_destroy(&indexes);
}

void test_common_avg_pooling(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    matrix_type m_values[] = {
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
    TEST_ASSERT_MATRIX_VALUE(4.25f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(3.5f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(6.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(4.f, matrix2d_get_elem(&result, 1, 1));

    matrix2d_destroy(&result);
}

void test_common_matrix2d_reshape(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    matrix_type m_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
    };
    matrix2d_load(&m, 2, 3, &m_values[0]);

    matrix2d_reshape(&m, &result, 3, 2);
    TEST_ASSERT_EQUAL_INT(3, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_MATRIX_VALUE(4.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(3.f, matrix2d_get_elem(&result, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(8.f, matrix2d_get_elem(&result, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(9.f, matrix2d_get_elem(&result, 1, 1));
    TEST_ASSERT_MATRIX_VALUE(1.f, matrix2d_get_elem(&result, 2, 0));
    TEST_ASSERT_MATRIX_VALUE(2.f, matrix2d_get_elem(&result, 2, 1));

    matrix2d_destroy(&result);
}

void test_common_matrix2d_reshape_2(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    matrix_type m_values[] = {
        4.f, 3.f, 8.f,
        9.f, 1.f, 2.f,
        7.f, 7.f, 6.f
    };
    matrix2d_load(&m, 3, 3, &m_values[0]);

    matrix2d_reshape(&m, &result, 1, 9);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(9, result.cols_n);
    TEST_ASSERT_MATRIX_VALUE(4.f, matrix2d_get_elem(&result, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(1.f, matrix2d_get_elem(&result, 0, 4));
    TEST_ASSERT_MATRIX_VALUE(7.f, matrix2d_get_elem(&result, 0, 6));
    TEST_ASSERT_MATRIX_VALUE(6.f, matrix2d_get_elem(&result, 0, 8));

    matrix2d_destroy(&result);
}

void test_common_matrix3d_reshape(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    matrix_type m_values[] = {
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

    TEST_ASSERT_MATRIX_VALUE(4.f, matrix3d_get_elem(&result, 0, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(1.f, matrix3d_get_elem(&result, 0, 3, 0));
    TEST_ASSERT_MATRIX_VALUE(2.f, matrix3d_get_elem(&result, 0, 7, 0));
    TEST_ASSERT_MATRIX_VALUE(7.f, matrix3d_get_elem(&result, 0, 8, 0));
    TEST_ASSERT_MATRIX_VALUE(2.f, matrix3d_get_elem(&result, 0, 11, 0));

    matrix3d_destroy(&result);
}

void test_common_matrix3d_reshape_2(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    matrix_type m_values[] = {
        4.f, 3.f, 9.f, 1.f, 5.f, 2.f, 9.f, 2.f, 7.f, 1.f, 3.f, 2.f
    };

    matrix3d_load(&m, 1, 12, 1, &m_values[0]);
    matrix3d_init(&result, 3, 2, 2);

    matrix3d_reshape(&m, &result);
    TEST_ASSERT_MATRIX_VALUE(4.f, matrix3d_get_elem(&result, 0, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(3.f, matrix3d_get_elem(&result, 0, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(9.f, matrix3d_get_elem(&result, 1, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(2.f, matrix3d_get_elem(&result, 2, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(2.f, matrix3d_get_elem(&result, 0, 1, 1));
    TEST_ASSERT_MATRIX_VALUE(3.f, matrix3d_get_elem(&result, 2, 0, 1));

    matrix3d_destroy(&result);
}

void test_common_matrix2d_softmax_inplace(void){
    matrix2d_t m = {0};
    matrix_type m_values[] = {2.f, 1.f, 0.1f};
    
    matrix2d_load(&m, 1, 3, &m_values[0]);

    matrix2d_softmax_inplace(&m);
    TEST_ASSERT_MATRIX_VALUE(0.5016878, matrix2d_get_elem(&m, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(0.304289, matrix2d_get_elem(&m, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(0.1940232, matrix2d_get_elem(&m, 0, 2));
}

// void test_parse_line(void){
//     char str[] = "1,255,12,123,2";
//     int length = sizeof(str)/sizeof(char);
    
//     matrix2d_t res = {0};
//     matrix_type label = 0.f;

//     matrix2d_init(&res, 2, 2);
//     parse_line(str, length, &res, &label);
//     TEST_ASSERT_MATRIX_VALUE(1.f, label);
//     TEST_ASSERT_MATRIX_VALUE(255.f, matrix2d_get_elem(&res, 0, 0));
//     TEST_ASSERT_MATRIX_VALUE(12.f, matrix2d_get_elem(&res, 0, 1));
//     TEST_ASSERT_MATRIX_VALUE(123.f, matrix2d_get_elem(&res, 1, 0));
//     TEST_ASSERT_MATRIX_VALUE(2.f, matrix2d_get_elem(&res, 1, 1));

//     matrix2d_destroy(&res);
// }

void test_zero_pad(void){
    matrix2d_t m = {0};
    matrix2d_t res = {0};

    matrix_type m_values[] = {
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
    matrix_type data[] = {1.f, 2.f, 3.f, 4.f};
    matrix_type* ptr = &data[0];

    matrix2d_load(&m, height, width, ptr);
    TEST_ASSERT_MATRIX_VALUE(data[0], matrix2d_get_elem(&m, 0, 0));
    TEST_ASSERT_MATRIX_VALUE(data[1], matrix2d_get_elem(&m, 0, 1));
    TEST_ASSERT_MATRIX_VALUE(data[2], matrix2d_get_elem(&m, 1, 0));
    TEST_ASSERT_MATRIX_VALUE(data[3], matrix2d_get_elem(&m, 1, 1));
}

#else
// TODO
void test_matrix2d_get_elem_as_ref(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix2d_get_elem_as_mut_ref(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix2d_get_elem(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix2d_set_elem(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix3d_get_elem_as_ref(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix3d_get_elem_as_mut_ref(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix3d_get_elem(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix3d_get_slice_as_mut_ref(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_cross_correlation_nopadding(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_cross_correlation_padding(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_cross_correlation_nopadding_stride(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_cross_correlation_padding_stride(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_max_pooling(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_avg_pooling(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_matrix2d_reshape(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_matrix2d_reshape_2(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_matrix3d_reshape(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_matrix3d_reshape_2(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_common_matrix2d_softmax_inplace(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_zero_pad(void){
    TEST_ASSERT_TRUE(true);
}

// TODO
void test_matrix2d_load(void){
    TEST_ASSERT_TRUE(true);
}

#endif

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