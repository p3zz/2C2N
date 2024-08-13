#ifndef __COMMON_H__
#define __COMMON_H__

typedef struct{
    int rows_n;
    int cols_n;
    float** values;
} matrix_t;

void cross_correlation(const matrix_t* const m1, const matrix_t* const m2, matrix_t* result);
matrix_t create_matrix(int rows_n, int cols_n);

#endif