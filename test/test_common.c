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

void test_common_cross_correlation(void){
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix_t m1 = create_matrix(3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix_t m2 = create_matrix(2, 2);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }
    matrix_t result = {};
    cross_correlation(&m1, &m2, &result);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(74.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[1][1]);
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_common_cross_correlation);

    int result = UNITY_END();

    return result;
}