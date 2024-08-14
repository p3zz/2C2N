#include <unity.h>
#include <stdbool.h>
#include "common.h"
#include "string.h"

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
    matrix2d_t m1 = create_matrix(3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = create_matrix(2, 2);
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
}

void test_common_cross_correlation_padding(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = create_matrix(3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = create_matrix(2, 2);
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
}

void test_common_cross_correlation_nopadding_stride(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = create_matrix(3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = create_matrix(2, 2);
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
}

void test_common_cross_correlation_padding_stride(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_t m1 = create_matrix(3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_t m2 = create_matrix(2, 2);
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
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_common_cross_correlation_nopadding);
    RUN_TEST(test_common_cross_correlation_padding);
    RUN_TEST(test_common_cross_correlation_nopadding_stride);
    RUN_TEST(test_common_cross_correlation_padding_stride);

    int result = UNITY_END();

    return result;
}