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

void test_always_true(void){
    TEST_ASSERT_TRUE(true);
}

void test_common_cross_correlation_nopadding(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m1, 3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_init(&m2, 2, 2);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }

    int padding = 0;
    int stride = 1;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(74.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[1][1]);

    matrix2d_destroy(&m1);
    matrix2d_destroy(&m2);
    matrix2d_destroy(&result);
}

void test_common_cross_correlation_padding(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m1, 3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_init(&m2, 2, 2);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }

    int padding = 1;
    int stride = 1;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
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
    
    matrix2d_destroy(&m1);
    matrix2d_destroy(&m2);
    matrix2d_destroy(&result);
}

void test_common_cross_correlation_nopadding_stride(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};
    
    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m1, 3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_init(&m2, 2, 2);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }
    
    int padding = 0;
    int stride = 2;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(1, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(54.f, result.values[0][0]);

    matrix2d_destroy(&m1);
    matrix2d_destroy(&m2);
    matrix2d_destroy(&result);
}

void test_common_cross_correlation_padding_stride(void){
    matrix2d_t m1 = {0};
    matrix2d_t m2 = {0};
    matrix2d_t result = {};

    const float m1_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m1, 3, 3);
    for(int i=0;i<m1.rows_n;i++){
        for(int j=0;j<m1.cols_n;j++){
            m1.values[i][j] = m1_values[i][j];
        }
    }
    
    const float m2_values[2][2] = {
        {4, 3},
        {3, 2}
    };
    matrix2d_init(&m2, 2, 2);
    for(int i=0;i<m2.rows_n;i++){
        for(int j=0;j<m2.cols_n;j++){
            m2.values[i][j] = m2_values[i][j];
        }
    }

    int padding = 1;
    int stride = 2;
    int result_height;
    int result_width;
    compute_output_size(m1.rows_n, m1.cols_n, m2.rows_n, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);
    full_cross_correlation(&m1, &m2, &result, padding, stride);
    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(25.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(41.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(43.f, result.values[1][1]);

    matrix2d_destroy(&m1);
    matrix2d_destroy(&m2);
    matrix2d_destroy(&result);
}

void test_common_max_pooling(void){
    matrix2d_t m = {0};
    matrix2d_t result = {};
    matrix3d_t indexes = {0};

    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m, 3, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }

    int kernel_size = 2;
    int padding = 0;
    int stride = 1;

    int result_height;
    int result_width;
    compute_output_size(m.rows_n, m.cols_n, kernel_size, padding, stride, &result_height, &result_width);
    matrix2d_init(&result, result_height, result_width);

    matrix3d_init(&indexes, result_height, result_width, 2);

    max_pooling(&m, &result, &indexes, kernel_size, padding, stride);

    TEST_ASSERT_EQUAL_INT(2, indexes.layers[0].rows_n);
    TEST_ASSERT_EQUAL_INT(2, indexes.layers[0].cols_n);

    TEST_ASSERT_EQUAL_INT(2, indexes.layers[1].rows_n);
    TEST_ASSERT_EQUAL_INT(2, indexes.layers[1].cols_n);

    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);

    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_INT(1, indexes.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_INT(0, indexes.layers[1].values[0][0]);

    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_INT(0, indexes.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_INT(2, indexes.layers[1].values[0][1]);

    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_INT(1, indexes.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_INT(0, indexes.layers[1].values[1][0]);

    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[1][1]);
    TEST_ASSERT_EQUAL_INT(2, indexes.layers[0].values[1][1]);
    TEST_ASSERT_EQUAL_INT(1, indexes.layers[1].values[1][1]);

    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
    matrix3d_destroy(&indexes);
}

void test_common_avg_pooling(void){
    matrix2d_t m = {0};
    matrix2d_t result = {};

    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m, 3, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }

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
    TEST_ASSERT_EQUAL_FLOAT(4.25f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.5f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[1][1]);

    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
}

void test_common_matrix2d_rotate180(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m, 3, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_rotate180(&m, &result);
    TEST_ASSERT_EQUAL_INT(3, result.rows_n);
    TEST_ASSERT_EQUAL_INT(3, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[0][2]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[1][2]);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[2][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.f, result.values[2][1]);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[2][2]);
    
    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
}

void test_common_matrix3d_submatrix(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m, 3, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_submatrix(&m, &result, 1, 2, 1, 2);

    TEST_ASSERT_EQUAL_INT(2, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[1][1]);

    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
}

void test_common_matrix3d_submatrix_2(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m, 3, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_submatrix(&m, &result, 0, 0, 1, 1);

    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(1, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(3.f, result.values[0][0]);

    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
}

void test_common_matrix2d_reshape(void){
    matrix2d_t m = {0};
    matrix2d_t result = {0};

    const float m_values[2][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
    };
    matrix2d_init(&m, 2, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_reshape(&m, &result, 3, 2);
    TEST_ASSERT_EQUAL_INT(3, result.rows_n);
    TEST_ASSERT_EQUAL_INT(2, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.f, result.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(8.f, result.values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(9.f, result.values[1][1]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.values[2][0]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.values[2][1]);

    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
}

void test_common_matrix2d_reshape_2(void){
    matrix2d_t m = {0};
    matrix2d_t result = {};

    const float m_values[3][3] = {
        {4.f, 3.f, 8.f},
        {9.f, 1.f, 2.f},
        {7.f, 7.f, 6.f}
    };
    matrix2d_init(&m, 3, 3);
    for(int i=0;i<m.rows_n;i++){
        for(int j=0;j<m.cols_n;j++){
            m.values[i][j] = m_values[i][j];
        }
    }
    matrix2d_reshape(&m, &result, 1, 9);
    TEST_ASSERT_EQUAL_INT(1, result.rows_n);
    TEST_ASSERT_EQUAL_INT(9, result.cols_n);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.values[0][4]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.values[0][6]);
    TEST_ASSERT_EQUAL_FLOAT(6.f, result.values[0][8]);

    matrix2d_destroy(&m);
    matrix2d_destroy(&result);
}

void test_common_matrix3d_reshape(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    const float m_values[2][3][2] = {
        {
            {4.f, 3.f},
            {9.f, 1.f},
            {5.f, 2.f},
        },
        {
            {9.f, 2.f},
            {7.f, 1.f},
            {3.f, 2.f},
        }
    };
    matrix3d_init(&m, 3, 2, 2);
    for(int i=0;i<m.depth;i++){
        for(int j=0;j<m.layers[i].rows_n;j++){
            for(int k=0;k<m.layers[i].cols_n;k++){
                m.layers[i].values[j][k] = m_values[i][j][k];                
            }    
        }
    }
    matrix3d_print(&m);
    matrix3d_init(&result, 1, 12, 1);
    matrix3d_reshape(&m, &result);
    matrix3d_print(&result);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, result.layers[0].values[0][3]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.layers[0].values[0][7]);
    TEST_ASSERT_EQUAL_FLOAT(7.f, result.layers[0].values[0][8]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.layers[0].values[0][11]);

    matrix3d_destroy(&m);
    matrix3d_destroy(&result);
}

void test_common_matrix3d_reshape_2(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    const float m_values[1][1][12] = {
        {
            {4.f, 3.f, 9.f, 1.f, 5.f, 2.f, 9.f, 2.f, 7.f, 1.f, 3.f, 2.f}
        }
    };
    matrix3d_init(&m, 1, 12, 1);
    for(int i=0;i<m.depth;i++){
        for(int j=0;j<m.layers[i].rows_n;j++){
            for(int k=0;k<m.layers[i].cols_n;k++){
                m.layers[i].values[j][k] = m_values[i][j][k];                
            }    
        }
    }
    matrix3d_print(&m);
    matrix3d_init(&result, 3, 2, 2);
    matrix3d_reshape(&m, &result);
    matrix3d_print(&result);
    TEST_ASSERT_EQUAL_FLOAT(4.f, result.layers[0].values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(3.f, result.layers[0].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(9.f, result.layers[0].values[1][0]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.layers[0].values[2][1]);
    TEST_ASSERT_EQUAL_FLOAT(2.f, result.layers[1].values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(3.f, result.layers[1].values[2][0]);

    matrix3d_destroy(&m);
    matrix3d_destroy(&result);
}

void test_matrix2d_softmax_inplace(void){
    matrix2d_t m = {0};

    const float m_values[3] = {2.f, 1.f, 0.1f};
    matrix2d_init(&m, 1, 3);
    for(int i=0;i<3;i++){
        m.values[0][i] = m_values[i];
    }
    matrix2d_softmax_inplace(&m);
    TEST_ASSERT_EQUAL_FLOAT(0.659001, m.values[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(0.242432, m.values[0][1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0985659, m.values[0][2]);

    matrix2d_destroy(&m);
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
    RUN_TEST(test_common_matrix2d_rotate180);
    RUN_TEST(test_common_matrix3d_submatrix);
    RUN_TEST(test_common_matrix3d_submatrix_2);
    RUN_TEST(test_common_matrix2d_reshape);
    RUN_TEST(test_common_matrix2d_reshape_2);
    RUN_TEST(test_common_matrix3d_reshape);
    RUN_TEST(test_common_matrix3d_reshape_2);
    RUN_TEST(test_matrix2d_softmax_inplace);
    int result = UNITY_END();

    return result;
}