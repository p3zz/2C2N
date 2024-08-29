#include <unity.h>
#include <stdbool.h>
#include "common.h"
#include "string.h"
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

void test_common_cross_correlation_nopadding(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = {0};
    create_matrix2d(&m1, 3, 3, true);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = {0};
    create_matrix2d(&m2, 2, 2, true);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }
    matrix2d_t result = {};
    cross_correlation(&m1, &m2, &result, 0, 1);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(74.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[1][1]);

    destroy_matrix2d(&m1);
    destroy_matrix2d(&m2);
    destroy_matrix2d(&result);
}

void test_common_cross_correlation_padding(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = {0};
    create_matrix2d(&m1, 3, 3, true);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = {0};
    create_matrix2d(&m2, 2, 2, true);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }
    matrix2d_t result = {};
    cross_correlation(&m1, &m2, &result, 1, 1);
    TEST_ASSERT_EQUAL_INT(4, result.rows_n);
    TEST_ASSERT_EQUAL_INT(4, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(18.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(25.f, result.values[0][2]);
    TEST_ASSERT_EQUAL_FLOAT(24.f, result.values[0][3]);

    TEST_ASSERT_EQUAL_FLOAT(30.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(54.f, result.values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[1][2]);
    TEST_ASSERT_EQUAL_FLOAT(38.f, result.values[1][3]);

    TEST_ASSERT_EQUAL_FLOAT(41.f, result.values[2][0]);
    TEST_ASSERT_EQUAL_FLOAT(74.f, result.values[2][1]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[2][2]);
    TEST_ASSERT_EQUAL_FLOAT(26.f, result.values[2][3]);

    TEST_ASSERT_EQUAL_FLOAT(21.f, result.values[3][0]);
    TEST_ASSERT_EQUAL_FLOAT(49.f, result.values[3][1]);
    TEST_ASSERT_EQUAL_FLOAT(46.f, result.values[3][2]);
    TEST_ASSERT_EQUAL_FLOAT(24.f, result.values[3][3]);
    
    destroy_matrix2d(&m1);
    destroy_matrix2d(&m2);
    destroy_matrix2d(&result);
}

void test_common_cross_correlation_nopadding_stride(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = {0};
    create_matrix2d(&m1, 3, 3, true);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = {0};
    create_matrix2d(&m2, 2, 2, true);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }
    matrix2d_t result = {};
    cross_correlation(&m1, &m2, &result, 0, 2);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(1, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, result.values[0][0]);

    destroy_matrix2d(&m1);
    destroy_matrix2d(&m2);
    destroy_matrix2d(&result);
}

void test_common_cross_correlation_padding_stride(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = {0};
    create_matrix2d(&m1, 3, 3, true);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = {0};
    create_matrix2d(&m2, 2, 2, true);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }
    matrix2d_t result = {};
    cross_correlation(&m1, &m2, &result, 1, 2);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(25.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(41.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[1][1]);

    destroy_matrix2d(&m1);
    destroy_matrix2d(&m2);
    destroy_matrix2d(&result);
}

void test_common_max_pooling(void){
    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m = {0};
    create_matrix2d(&m, 3, 3, true);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_t result = {};
    max_pooling(&m, &result, 2, 0, 1);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[1][1]);

    destroy_matrix2d(&m);
    destroy_matrix2d(&result);
}

void test_common_avg_pooling(void){
    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m = {0};
    create_matrix2d(&m, 3, 3, true);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_t result = {};
    avg_pooling(&m, &result, 2, 0, 1);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.25f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.5f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[1][1]);

    destroy_matrix2d(&m);
    destroy_matrix2d(&result);
}

void test_common_matrix2d_flatten(void){
    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m = {0};
    create_matrix2d(&m, 3, 3, true);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_t result = {};
    matrix2d_flatten(&m, &result);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(9, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.values[0][4]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[0][6]);
    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[0][8]);
}

void test_common_matrix2d_rotate180(void){
    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m = {0};
    create_matrix2d(&m, 3, 3, false);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_t result = {0};
    matrix2d_rotate180(&m, &result);

    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[0][2]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[1][2]);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[2][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.f, result.values[2][1]);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[2][2]);
    
    destroy_matrix2d(&m);
    destroy_matrix2d(&result);
}

int main(void)
{
    srand(0);
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_common_cross_correlation_nopadding);
    RUN_TEST(test_common_cross_correlation_padding);
    RUN_TEST(test_common_cross_correlation_nopadding_stride);
    RUN_TEST(test_common_cross_correlation_padding_stride);
    RUN_TEST(test_common_max_pooling);
    RUN_TEST(test_common_avg_pooling);
    RUN_TEST(test_common_matrix2d_flatten);
    RUN_TEST(test_common_matrix2d_rotate180);
    int result = UNITY_END();

    return result;
}