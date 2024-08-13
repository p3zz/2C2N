#include "common.h"
#include "stdlib.h"

void cross_correlation(const matrix_t* const m1, const matrix_t* const m2, matrix_t* result){
    int output_rows = m1->rows_n - m2->rows_n + 1;
    int output_cols = m1->cols_n - m2->cols_n + 1;

    *result = create_matrix(output_rows, output_cols);

    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float sum = 0;
            for(int m=0;m<m2->rows_n;m++){
                for(int n=0;n<m2->cols_n;n++){
                    // sum += matrix1[(i + m) * cols1 + (j + n)] * matrix2[m * cols2 + n];
                    sum += (m1->values[i + m][j + n]) * m2->values[m][n];
                }
            }
            result->values[i][j] = sum;
        }
    }
}

matrix_t create_matrix(int rows_n, int cols_n){
    matrix_t m = {0};
    m.rows_n = rows_n;
    m.cols_n = cols_n;
    m.values = (float**)malloc(rows_n * sizeof(float*));
    for(int i=0;i<rows_n;i++){
        m.values[i] = (float*)malloc(cols_n * sizeof(float));
    }
    return m;
}